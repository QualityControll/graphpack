use proc_macro::TokenStream;
use quote::{format_ident, quote, ToTokens};
use syn::{
    parse_macro_input, BinOp, Expr, ExprBinary, ExprBlock, ExprClosure, ExprLit, ExprPath,
    GenericArgument, Local, Pat, PathArguments, Stmt, Type, UnOp,
};

#[derive(Clone, Copy)]
enum ScalarType {
    F32,
    I32,
}

impl ScalarType {
    fn data_type(self) -> proc_macro2::TokenStream {
        match self {
            Self::F32 => quote!(::tensorflow::DataType::Float),
            Self::I32 => quote!(::tensorflow::DataType::Int32),
        }
    }

    fn tensor_type(self) -> proc_macro2::TokenStream {
        match self {
            Self::F32 => quote!(f32),
            Self::I32 => quote!(i32),
        }
    }
}

struct Lowerer {
    scalar_type: ScalarType,
    next_id: usize,
}

impl Lowerer {
    fn next_name(&mut self, prefix: &str) -> syn::Ident {
        let id = self.next_id;
        self.next_id += 1;
        format_ident!("__graphpack_{}_{}", prefix, id)
    }

    fn lower_expr(&mut self, expr: &Expr) -> syn::Result<proc_macro2::TokenStream> {
        match expr {
            Expr::Path(ExprPath { path, .. }) if path.segments.len() == 1 => {
                let ident = &path.segments[0].ident;
                Ok(quote!(#ident.clone()))
            }
            Expr::Lit(ExprLit { lit, .. }) => self.lower_literal(lit),
            Expr::Binary(ExprBinary { left, op, right, .. }) => {
                self.lower_binary(left, op, right)
            }
            Expr::Unary(expr) => self.lower_unary(&expr.op, &expr.expr),
            Expr::Block(ExprBlock { block, .. }) => self.lower_block(block),
            _ => Err(syn::Error::new_spanned(
                expr,
                "graphpack! currently supports variables, scalar literals, arithmetic/bitwise operators, and blocks",
            )),
        }
    }

    fn lower_literal(&mut self, lit: &syn::Lit) -> syn::Result<proc_macro2::TokenStream> {
        let tensor_type = self.scalar_type.tensor_type();
        let name = self.next_name("const");
        let value = match (self.scalar_type, lit) {
            (ScalarType::F32, syn::Lit::Float(value)) => value.to_token_stream(),
            (ScalarType::F32, syn::Lit::Int(value)) => value.to_token_stream(),
            (ScalarType::I32, syn::Lit::Int(value)) => value.to_token_stream(),
            _ => {
                return Err(syn::Error::new_spanned(
                    lit,
                    "literal type does not match the graph input type",
                ))
            }
        };

        Ok(quote! {
            {
                let mut __op = graph
                    .new_operation("Const", stringify!(#name))
                    .expect("failed to create Const operation");
                __op
                    .set_attr_type("dtype", #self_scalar_data_type)
                    .expect("failed to set Const dtype");
                let __value = ::tensorflow::Tensor::<#tensor_type>::from(#value);
                __op
                    .set_attr_tensor("value", __value)
                    .expect("failed to set Const value");
                __op.finish().expect("failed to finish Const operation").into()
            }
        })
    }

    fn lower_binary(
        &mut self,
        left: &Expr,
        op: &BinOp,
        right: &Expr,
    ) -> syn::Result<proc_macro2::TokenStream> {
        let op_name = match op {
            BinOp::Add(_) => "Add",
            BinOp::Sub(_) => "Sub",
            BinOp::Mul(_) => "Mul",
            BinOp::Div(_) => "Div",
            BinOp::Rem(_) => "FloorMod",
            BinOp::BitAnd(_) => "BitwiseAnd",
            BinOp::BitOr(_) => "BitwiseOr",
            BinOp::BitXor(_) => "BitwiseXor",
            BinOp::Shl(_) => "LeftShift",
            BinOp::Shr(_) => "RightShift",
            _ => {
                return Err(syn::Error::new_spanned(
                    op,
                    "unsupported binary operator in graphpack!",
                ))
            }
        };

        let left = self.lower_expr(left)?;
        let right = self.lower_expr(right)?;
        let name = self.next_name("op");

        Ok(quote! {
            {
                let mut __op = graph
                    .new_operation(#op_name, stringify!(#name))
                    .expect("failed to create TensorFlow operation");
                __op.add_input(#left);
                __op.add_input(#right);
                __op.finish().expect("failed to finish TensorFlow operation").into()
            }
        })
    }

    fn lower_unary(
        &mut self,
        op: &UnOp,
        expr: &Expr,
    ) -> syn::Result<proc_macro2::TokenStream> {
        let op_name = match op {
            UnOp::Neg(_) => "Neg",
            UnOp::Not(_) => "Invert",
            _ => return Err(syn::Error::new_spanned(op, "unsupported unary operator in graphpack!")),
        };
        let expr = self.lower_expr(expr)?;
        let name = self.next_name("unary");

        Ok(quote! {
            {
                let mut __op = graph
                    .new_operation(#op_name, stringify!(#name))
                    .expect("failed to create TensorFlow operation");
                __op.add_input(#expr);
                __op.finish().expect("failed to finish TensorFlow operation").into()
            }
        })
    }

    fn lower_block(&mut self, block: &syn::Block) -> syn::Result<proc_macro2::TokenStream> {
        let mut statements = Vec::new();
        let mut final_expr = None;

        for (index, stmt) in block.stmts.iter().enumerate() {
            let is_last = index + 1 == block.stmts.len();
            match stmt {
                Stmt::Local(local) => statements.push(self.lower_local(local)?),
                Stmt::Expr(expr, semi) => {
                    let lowered = self.lower_expr(expr)?;
                    if is_last && semi.is_none() {
                        final_expr = Some(lowered);
                    } else {
                        statements.push(quote! { let _ = #lowered; });
                    }
                }
                Stmt::Item(item) => {
                    return Err(syn::Error::new_spanned(
                        item,
                        "items are not supported inside graphpack! closures",
                    ))
                }
                Stmt::Macro(mac) => {
                    return Err(syn::Error::new_spanned(
                        mac,
                        "macros are not supported inside graphpack! closures",
                    ))
                }
            }
        }

        let result = final_expr.unwrap_or_else(|| quote! { ::tensorflow::Output::from(input_operation.clone()) });
        Ok(quote! {{ #(#statements)* #result }})
    }

    fn lower_local(&mut self, local: &Local) -> syn::Result<proc_macro2::TokenStream> {
        let ident = match &local.pat {
            Pat::Ident(pat_ident) => &pat_ident.ident,
            _ => return Err(syn::Error::new_spanned(&local.pat, "graphpack! let bindings must use identifiers")),
        };
        let init = local.init.as_ref().ok_or_else(|| {
            syn::Error::new_spanned(local, "graphpack! let bindings require an initializer")
        })?;
        let expr = self.lower_expr(&init.expr)?;
        Ok(quote! { let #ident = #expr; })
    }
}

#[proc_macro]
pub fn graphpack(input: TokenStream) -> TokenStream {
    let closure = parse_macro_input!(input as ExprClosure);

    if closure.inputs.len() != 1 {
        return syn::Error::new_spanned(closure, "graphpack! currently supports exactly one input")
            .to_compile_error()
            .into();
    }

    let input = closure.inputs.first().unwrap();
    let (input_name, scalar_type) = match input {
        Pat::Type(pat_type) => {
            let name = match &*pat_type.pat {
                Pat::Ident(pat_ident) => pat_ident.ident.clone(),
                _ => return syn::Error::new_spanned(&pat_type.pat, "graphpack! inputs must be identifiers").to_compile_error().into(),
            };
            let scalar_type = match &*pat_type.ty {
                Type::Path(type_path) => {
                    let segment = type_path.path.segments.last().unwrap();
                    if segment.ident != "Input" {
                        return syn::Error::new_spanned(&pat_type.ty, "graphpack! closure inputs must have type Input<T>").to_compile_error().into();
                    }
                    match &segment.arguments {
                        PathArguments::AngleBracketed(arguments) if arguments.args.len() == 1 => match arguments.args.first().unwrap() {
                            GenericArgument::Type(Type::Path(type_path)) if type_path.path.is_ident("f32") => ScalarType::F32,
                            GenericArgument::Type(Type::Path(type_path)) if type_path.path.is_ident("i32") => ScalarType::I32,
                            other => return syn::Error::new_spanned(other, "graphpack! currently supports Input<f32> and Input<i32>").to_compile_error().into(),
                        },
                        _ => return syn::Error::new_spanned(&segment.arguments, "graphpack! inputs must have type Input<T>").to_compile_error().into(),
                    }
                }
                _ => return syn::Error::new_spanned(&pat_type.ty, "graphpack! inputs must have type Input<T>").to_compile_error().into(),
            };
            (name, scalar_type)
        }
        _ => return syn::Error::new_spanned(input, "graphpack! inputs must have the form |x: Input<T>| ...").to_compile_error().into(),
    };

    let mut lowerer = Lowerer { scalar_type, next_id: 0 };
    let body = match lowerer.lower_expr(&closure.body) {
        Ok(body) => body,
        Err(error) => return error.to_compile_error().into(),
    };
    let data_type = scalar_type.data_type();

    TokenStream::from(quote! {
        {
            let mut graph = ::tensorflow::Graph::new();
            let mut input_operation = graph
                .new_operation("Placeholder", stringify!(#input_name))
                .expect("failed to create input operation");
            input_operation
                .set_attr_type("dtype", #data_type)
                .expect("failed to set input data type");
            let input_operation = input_operation
                .finish()
                .expect("failed to finish input operation");
            let __graphpack_result = #body;
            let mut __graphpack_output = graph
                .new_operation("Identity", "output")
                .expect("failed to create output operation");
            __graphpack_output.add_input(__graphpack_result);
            __graphpack_output
                .finish()
                .expect("failed to finish output operation");
            graph.graph_def().expect("failed to serialize GraphDef")
        }
    })
}

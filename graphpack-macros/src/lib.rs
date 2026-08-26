use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, BinOp, Expr, ExprBinary, ExprClosure, GenericArgument, Pat, PathArguments, Type};

#[proc_macro]
pub fn graphpack(input: TokenStream) -> TokenStream {
    let closure = parse_macro_input!(input as ExprClosure);

    if closure.inputs.len() != 1 {
        return syn::Error::new_spanned(closure, "graphpack! currently supports exactly one input")
            .to_compile_error()
            .into();
    }

    let input = closure.inputs.first().unwrap();
    let (input_name, data_type) = match input {
        Pat::Type(pat_type) => {
            let name = match &*pat_type.pat {
                Pat::Ident(pat_ident) => pat_ident.ident.clone(),
                _ => {
                    return syn::Error::new_spanned(
                        &pat_type.pat,
                        "graphpack! inputs must be identifiers",
                    )
                    .to_compile_error()
                    .into()
                }
            };
            let data_type = match &*pat_type.ty {
                Type::Path(type_path) => {
                    let segment = type_path.path.segments.last().unwrap();
                    if segment.ident != "Input" {
                        return syn::Error::new_spanned(
                            &pat_type.ty,
                            "graphpack! closure inputs must have type Input<T>",
                        )
                        .to_compile_error()
                        .into();
                    }
                    match &segment.arguments {
                        PathArguments::AngleBracketed(arguments) if arguments.args.len() == 1 => {
                            match arguments.args.first().unwrap() {
                                GenericArgument::Type(Type::Path(type_path))
                                    if type_path.path.is_ident("f32") =>
                                {
                                    quote!(::tensorflow::DataType::Float)
                                }
                                other => {
                                    return syn::Error::new_spanned(
                                        other,
                                        "graphpack! currently supports Input<f32>",
                                    )
                                    .to_compile_error()
                                    .into()
                                }
                            }
                        }
                        _ => {
                            return syn::Error::new_spanned(
                                &segment.arguments,
                                "graphpack! inputs must have type Input<T>",
                            )
                            .to_compile_error()
                            .into()
                        }
                    }
                }
                _ => {
                    return syn::Error::new_spanned(
                        &pat_type.ty,
                        "graphpack! inputs must have type Input<T>",
                    )
                    .to_compile_error()
                    .into()
                }
            };
            (name, data_type)
        }
        _ => {
            return syn::Error::new_spanned(
                input,
                "graphpack! inputs must have the form |x: Input<T>| x + 1.0",
            )
            .to_compile_error()
            .into()
        }
    };

    let body = match closure.body.as_ref() {
        Expr::Binary(ExprBinary {
            left,
            op: BinOp::Add(_),
            right,
            ..
        }) => {
            if !matches!(left.as_ref(), Expr::Path(path) if path.path.is_ident(&input_name)) {
                return syn::Error::new_spanned(left, "left operand must be the graph input")
                    .to_compile_error()
                    .into();
            }
            let value = match right.as_ref() {
                Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Float(lit),
                    ..
                }) => lit.clone(),
                _ => {
                    return syn::Error::new_spanned(right, "right operand must be an f32 literal")
                        .to_compile_error()
                        .into()
                }
            };
            quote! {
                let mut constant = graph
                    .new_operation("Const", "constant")
                    .expect("failed to create constant operation");
                constant
                    .set_attr_type("dtype", ::tensorflow::DataType::Float)
                    .expect("failed to set constant data type");
                let value = ::tensorflow::Tensor::<f32>::new(&[])
                    .with_values(&[#value as f32])
                    .expect("failed to create constant tensor");
                constant
                    .set_attr_tensor("value", value)
                    .expect("failed to set constant value");
                let constant = constant
                    .finish()
                    .expect("failed to finish constant operation");
                let mut add = graph
                    .new_operation("Add", "output")
                    .expect("failed to create Add operation");
                add.add_input(input_operation);
                add.add_input(constant);
                add.finish().expect("failed to finish Add operation");
            }
        }
        _ => {
            return syn::Error::new_spanned(
                &closure.body,
                "graphpack! currently supports |x: Input<f32>| x + 1.0",
            )
            .to_compile_error()
            .into()
        }
    };

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
            #body
            graph.graph_def().expect("failed to serialize GraphDef")
        }
    })
}

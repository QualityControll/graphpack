use proc_macro2::{Ident, TokenStream};
use quote::quote;
use std::collections::HashMap;
use syn::{Expr, ExprBinary, ExprLit, ExprPath, ExprUnary, Lit, Stmt};

use crate::ops;
use crate::types::ScalarType;

pub struct LoweringContext {
    pub values: HashMap<String, TokenStream>,
    pub scalar_type: ScalarType,
}

impl LoweringContext {
    pub fn new(scalar_type: ScalarType) -> Self {
        Self {
            values: HashMap::new(),
            scalar_type,
        }
    }

    pub fn lower_expr(&mut self, expr: &Expr) -> syn::Result<TokenStream> {
        match expr {
            Expr::Path(ExprPath { path, .. }) => {
                let name = path.segments.last().unwrap().ident.to_string();
                self.values
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| syn::Error::new_spanned(expr, "unknown graph value"))
            }
            Expr::Lit(ExprLit {
                lit: Lit::Float(value),
                ..
            }) => {
                let ty = self.scalar_type.tensor_type();
                let data_type = self.scalar_type.data_type();
                Ok(quote!({
                    let mut constant = graph
                        .new_operation("Const", "constant")
                        .expect("failed to create constant operation");
                    constant
                        .set_attr_type("dtype", #data_type)
                        .expect("failed to set constant dtype");
                    let value = ::tensorflow::Tensor::<#ty>::new(&[])
                        .with_values(&[#value as #ty])
                        .expect("failed to create constant tensor");
                    constant
                        .set_attr_tensor("value", value)
                        .expect("failed to set constant value");
                    constant
                        .finish()
                        .expect("failed to finish constant operation")
                }))
            }
            Expr::Binary(ExprBinary { left, op, right, .. }) => {
                let op_name = ops::binary_op(op)
                    .ok_or_else(|| syn::Error::new_spanned(op, "unsupported binary operator"))?;
                let left = self.lower_expr(left)?;
                let right = self.lower_expr(right)?;
                Ok(quote!({
                    let left_output = #left;
                    let right_output = #right;
                    let mut operation = graph
                        .new_operation(#op_name, "operation")
                        .expect("failed to create operation");
                    operation.add_input(left_output);
                    operation.add_input(right_output);
                    operation.finish().expect("failed to finish operation")
                }))
            }
            Expr::Unary(ExprUnary { op, expr, .. }) => {
                let op_name = ops::unary_op(op)
                    .ok_or_else(|| syn::Error::new_spanned(op, "unsupported unary operator"))?;
                let input = self.lower_expr(expr)?;
                Ok(quote!({
                    let input_output = #input;
                    let mut operation = graph
                        .new_operation(#op_name, "operation")
                        .expect("failed to create operation");
                    operation.add_input(input_output);
                    operation.finish().expect("failed to finish operation")
                }))
            }
            _ => Err(syn::Error::new_spanned(
                expr,
                "unsupported graph expression",
            )),
        }
    }

    pub fn lower_statements(&mut self, statements: &[Stmt]) -> syn::Result<TokenStream> {
        let mut generated = TokenStream::new();
        let mut final_expr = None;
        for statement in statements {
            match statement {
                Stmt::Local(local) => {
                    let ident = match &local.pat {
                        syn::Pat::Ident(pat) => pat.ident.clone(),
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &local.pat,
                                "graphpack! let bindings must be identifiers",
                            ))
                        }
                    };
                    let init = local.init.as_ref().ok_or_else(|| {
                        syn::Error::new_spanned(local, "graphpack! let bindings require an initializer")
                    })?;
                    let value = self.lower_expr(&init.expr)?;
                    self.values.insert(ident.to_string(), value.clone());
                    generated.extend(quote!(let #ident = #value;));
                }
                Stmt::Expr(expr, _) => final_expr = Some(self.lower_expr(expr)?),
                Stmt::Item(item) => {
                    return Err(syn::Error::new_spanned(
                        item,
                        "items are not supported in graphpack! closures",
                    ))
                }
                Stmt::Macro(mac) => {
                    return Err(syn::Error::new_spanned(
                        mac,
                        "macros are not supported in graphpack! closures",
                    ))
                }
            }
        }
        Ok(quote!(#generated #final_expr))
    }
}

pub fn input_ident(name: &str) -> Ident {
    syn::Ident::new(name, proc_macro2::Span::call_site())
}

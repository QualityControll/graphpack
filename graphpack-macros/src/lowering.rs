use proc_macro2::TokenStream;
use quote::quote;
use std::collections::HashMap;
use syn::{Expr, ExprBinary, ExprIf, ExprLit, ExprPath, ExprUnary, Lit, Stmt};

use crate::ops;
use crate::scalar_types::ScalarType;

pub struct LoweringContext {
    pub values: HashMap<String, TokenStream>,
    pub scalar_type: ScalarType,
    next_node_id: usize,
}

impl LoweringContext {
    pub fn new(scalar_type: ScalarType) -> Self { Self { values: HashMap::new(), scalar_type, next_node_id: 0 } }
    fn node_name(&mut self, prefix: &str) -> String { let id = self.next_node_id; self.next_node_id += 1; format!("{prefix}_{id}") }

    pub fn lower_expr(&mut self, expr: &Expr) -> syn::Result<TokenStream> {
        match expr {
            Expr::Path(ExprPath { path, .. }) => {
                let name = path.segments.last().unwrap().ident.to_string();
                self.values.get(&name).cloned().ok_or_else(|| syn::Error::new_spanned(expr, "unknown graph value"))
            }
            Expr::Lit(ExprLit { lit, .. }) => {
                let (value, ty, data_type) = match lit {
                    Lit::Float(value) => { let ty = self.scalar_type.tensor_type(); let data_type = self.scalar_type.data_type(); (quote!(#value as #ty), ty, data_type) }
                    Lit::Int(value) => { let ty = self.scalar_type.tensor_type(); let data_type = self.scalar_type.data_type(); (quote!(#value as #ty), ty, data_type) }
                    _ => return Err(syn::Error::new_spanned(lit, "unsupported graph literal")),
                };
                let node_name = self.node_name("constant");
                Ok(quote!({ let mut constant = graph.new_operation("Const", #node_name).expect("failed to create constant operation"); constant.set_attr_type("dtype", #data_type).expect("failed to set constant dtype"); let value = ::tensorflow::Tensor::<#ty>::new(&[]).with_values(&[#value]).expect("failed to create constant tensor"); constant.set_attr_tensor("value", value).expect("failed to set constant value"); let operation = constant.finish().expect("failed to finish constant operation"); ::tensorflow::Output::from(operation) }))
            }
            Expr::Binary(ExprBinary { left, op, right, .. }) => {
                let op_name = ops::binary_op(op).ok_or_else(|| syn::Error::new_spanned(op, "unsupported binary operator"))?;
                let left = self.lower_expr(left)?; let right = self.lower_expr(right)?; let node_name = self.node_name("operation");
                Ok(quote!({ let left_output: ::tensorflow::Output = #left; let right_output: ::tensorflow::Output = #right; let mut operation = graph.new_operation(#op_name, #node_name).expect("failed to create operation"); operation.add_input(left_output); operation.add_input(right_output); let operation = operation.finish().expect("failed to finish operation"); ::tensorflow::Output::from(operation) }))
            }
            Expr::Unary(ExprUnary { op, expr, .. }) => {
                let op_name = ops::unary_op(op).ok_or_else(|| syn::Error::new_spanned(op, "unsupported unary operator"))?;
                let input = self.lower_expr(expr)?; let node_name = self.node_name("operation");
                Ok(quote!({ let input_output: ::tensorflow::Output = #input; let mut operation = graph.new_operation(#op_name, #node_name).expect("failed to create operation"); operation.add_input(input_output); let operation = operation.finish().expect("failed to finish operation"); ::tensorflow::Output::from(operation) }))
            }
            Expr::If(ExprIf { cond, then_branch, else_branch, .. }) => {
                let condition = self.lower_expr(cond)?; let outer_values = self.values.clone();
                let then_value = self.lower_statements(&then_branch.stmts)?; self.values = outer_values.clone();
                let else_branch = else_branch.as_ref().ok_or_else(|| syn::Error::new_spanned(expr, "graphpack! if expressions require an else branch"))?;
                let else_value = match &*else_branch.1 { Expr::Block(block) => self.lower_statements(&block.block.stmts)?, expr => self.lower_expr(expr)? };
                self.values = outer_values;
                let node_name = self.node_name("if");
                Ok(quote!({ let condition_output: ::tensorflow::Output = #condition; let then_output: ::tensorflow::Output = #then_value; let else_output: ::tensorflow::Output = #else_value; let mut operation = graph.new_operation("SelectV2", #node_name).expect("failed to create conditional operation"); operation.add_input(condition_output); operation.add_input(then_output); operation.add_input(else_output); let operation = operation.finish().expect("failed to finish conditional operation"); ::tensorflow::Output::from(operation) }))
            }
            Expr::Block(block) => self.lower_statements(&block.block.stmts),
            _ => Err(syn::Error::new_spanned(expr, "unsupported graph expression")),
        }
    }

    fn lower_if_statement(&mut self, expr: &syn::ExprIf) -> syn::Result<()> {
        self.lower_expr(&expr.cond)?;
        let outer_values = self.values.clone();
        self.lower_statements_discarding_value(&expr.then_branch.stmts)?;
        self.values = outer_values.clone();
        let else_branch = expr.else_branch.as_ref().ok_or_else(|| syn::Error::new_spanned(expr, "graphpack! if statements require an else branch"))?;
        match &*else_branch.1 { Expr::Block(block) => self.lower_statements_discarding_value(&block.block.stmts)?, other => { self.lower_expr(other)?; } }
        self.values = outer_values;
        Ok(())
    }

    fn lower_statements_discarding_value(&mut self, statements: &[Stmt]) -> syn::Result<()> {
        for statement in statements {
            match statement {
                Stmt::Local(local) => {
                    let ident = match &local.pat { syn::Pat::Ident(pat) => pat.ident.clone(), _ => return Err(syn::Error::new_spanned(&local.pat, "graphpack! let bindings must be identifiers")) };
                    let init = local.init.as_ref().ok_or_else(|| syn::Error::new_spanned(local, "graphpack! let bindings require an initializer"))?;
                    let value = self.lower_expr(&init.expr)?;
                    self.values.insert(ident.to_string(), quote!(#ident.clone()));
                    let _ = value;
                }
                Stmt::Expr(expr, _) => { if let Expr::If(if_expr) = expr { self.lower_if_statement(if_expr)?; } else { self.lower_expr(expr)?; } }
                Stmt::Item(item) => return Err(syn::Error::new_spanned(item, "items are not supported in graphpack! closures")),
                Stmt::Macro(mac) => return Err(syn::Error::new_spanned(mac, "macros are not supported in graphpack! closures")),
            }
        }
        Ok(())
    }

    pub fn lower_statements(&mut self, statements: &[Stmt]) -> syn::Result<TokenStream> {
        let mut generated = TokenStream::new(); let mut final_expr = None;
        for statement in statements {
            match statement {
                Stmt::Local(local) => {
                    let ident = match &local.pat { syn::Pat::Ident(pat) => pat.ident.clone(), _ => return Err(syn::Error::new_spanned(&local.pat, "graphpack! let bindings must be identifiers")) };
                    let init = local.init.as_ref().ok_or_else(|| syn::Error::new_spanned(local, "graphpack! let bindings require an initializer"))?;
                    let value = self.lower_expr(&init.expr)?; self.values.insert(ident.to_string(), quote!(#ident.clone()));
                    generated.extend(quote! { let #ident: ::tensorflow::Output = #value; });
                }
                Stmt::Expr(expr, semi) => {
                    if semi.is_some() { if let Expr::If(if_expr) = expr { self.lower_if_statement(if_expr)?; } else { self.lower_expr(expr)?; } }
                    else { final_expr = Some(self.lower_expr(expr)?); }
                }
                Stmt::Item(item) => return Err(syn::Error::new_spanned(item, "items are not supported in graphpack! closures")),
                Stmt::Macro(mac) => return Err(syn::Error::new_spanned(mac, "macros are not supported in graphpack! closures")),
            }
        }
        let final_expr = final_expr.ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "graphpack! closure must return a graph value"))?;
        Ok(quote!(#generated #final_expr))
    }
}

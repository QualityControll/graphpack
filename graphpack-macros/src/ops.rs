use proc_macro2::TokenStream;
use quote::quote;
use syn::BinOp;

pub fn binary_op(op: &BinOp) -> Option<TokenStream> {
    let name = match op {
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
        BinOp::Eq(_) => "Equal",
        BinOp::Ne(_) => "NotEqual",
        BinOp::Lt(_) => "Less",
        BinOp::Le(_) => "LessEqual",
        BinOp::Gt(_) => "Greater",
        BinOp::Ge(_) => "GreaterEqual",
        _ => return None,
    };
    Some(quote!(#name))
}

pub fn unary_op(op: &syn::UnOp) -> Option<TokenStream> {
    let name = match op {
        syn::UnOp::Neg(_) => "Neg",
        syn::UnOp::Not(_) => "Invert",
        _ => return None,
    };
    Some(quote!(#name))
}

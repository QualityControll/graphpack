use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ExprClosure, ReturnType};

#[proc_macro]
pub fn graphpack(input: TokenStream) -> TokenStream {
    let closure = parse_macro_input!(input as ExprClosure);

    if closure.inputs.is_empty() && matches!(closure.output, ReturnType::Default) {
        TokenStream::from(quote! {
            ::graphpack::GraphDef::new()
        })
    } else {
        syn::Error::new_spanned(
            closure,
            "graphpack! currently only supports unit closures",
        )
        .to_compile_error()
        .into()
    }
}

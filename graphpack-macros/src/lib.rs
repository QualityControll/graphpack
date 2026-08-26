use proc_macro::TokenStream;

#[proc_macro]
pub fn graphpack(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ExprClosure, Pat};

mod types;

#[proc_macro]
pub fn graphpack(input: TokenStream) -> TokenStream {
    let closure = parse_macro_input!(input as ExprClosure);
    if closure.inputs.len() != 1 {
        return syn::Error::new_spanned(closure, "graphpack! currently supports exactly one input")
            .to_compile_error().into();
    }
    let input = closure.inputs.first().unwrap();
    let (input_name, scalar_type) = match input {
        Pat::Type(pat_type) => {
            let name = match &*pat_type.pat {
                Pat::Ident(pat_ident) => pat_ident.ident.clone(),
                _ => return syn::Error::new_spanned(&pat_type.pat, "graphpack! inputs must be identifiers").to_compile_error().into(),
            };
            let scalar_type = match types::ScalarType::from_input_type(&pat_type.ty) {
                Ok(value) => value,
                Err(error) => return error.to_compile_error().into(),
            };
            (name, scalar_type)
        }
        _ => return syn::Error::new_spanned(input, "graphpack! inputs must have the form |x: Input<T>| ...").to_compile_error().into(),
    };

    let data_type = scalar_type.data_type();
    let input_name_string = input_name.to_string();

    TokenStream::from(quote! {
        {
            let mut graph = ::tensorflow::Graph::new();
            let mut input_operation = graph.new_operation("Placeholder", #input_name_string)
                .expect("failed to create input operation");
            input_operation.set_attr_type("dtype", #data_type)
                .expect("failed to set input data type");
            let input_operation = input_operation.finish()
                .expect("failed to finish input operation");
            let _ = input_operation;
            graph.graph_def().expect("failed to serialize GraphDef")
        }
    })
}

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ExprClosure, Pat};

mod lowering;
mod ops;
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

    let mut context = lowering::LoweringContext::new(scalar_type);
    context.values.insert(input_name.to_string(), quote!(input_operation));
    let body = match &*closure.body {
        syn::Expr::Block(block) => match context.lower_statements(&block.block.stmts) {
            Ok(body) => body,
            Err(error) => return error.to_compile_error().into(),
        },
        expr => match context.lower_expr(expr) {
            Ok(value) => quote!(#value),
            Err(error) => return error.to_compile_error().into(),
        },
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
            let output = { #body };
            let mut identity = graph.new_operation("Identity", "output")
                .expect("failed to create output operation");
            identity.add_input(output);
            identity.finish().expect("failed to finish output operation");
            graph.graph_def().expect("failed to serialize GraphDef")
        }
    })
}

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ExprClosure, GenericArgument, Pat, PathArguments, ReturnType, Type};

#[proc_macro]
pub fn graphpack(input: TokenStream) -> TokenStream {
    let closure = parse_macro_input!(input as ExprClosure);

    if !matches!(closure.output, ReturnType::Default) {
        return syn::Error::new_spanned(
            closure,
            "graphpack! currently only supports closures with no return value",
        )
        .to_compile_error()
        .into();
    }

    if closure.inputs.is_empty() {
        return TokenStream::from(quote! {
            ::tensorflow::Graph::new().graph_def().expect("failed to serialize GraphDef")
        });
    }

    if closure.inputs.len() != 1 {
        return syn::Error::new_spanned(
            closure,
            "graphpack! currently supports at most one input",
        )
        .to_compile_error()
        .into();
    }

    let input = closure.inputs.first().unwrap();
    let (name, data_type) = match input {
        Pat::Type(pat_type) => {
            let name = match &*pat_type.pat {
                Pat::Ident(pat_ident) => &pat_ident.ident,
                _ => {
                    return syn::Error::new_spanned(
                        &pat_type.pat,
                        "graphpack! inputs must be identifiers",
                    )
                    .to_compile_error()
                    .into();
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
                                    if type_path.path.is_ident("f32") => {
                                    quote!(::tensorflow::DataType::Float)
                                }
                                other => {
                                    return syn::Error::new_spanned(
                                        other,
                                        "graphpack! currently supports Input<f32>",
                                    )
                                    .to_compile_error()
                                    .into();
                                }
                            }
                        }
                        _ => {
                            return syn::Error::new_spanned(
                                &segment.arguments,
                                "graphpack! inputs must have type Input<T>",
                            )
                            .to_compile_error()
                            .into();
                        }
                    }
                }
                _ => {
                    return syn::Error::new_spanned(
                        &pat_type.ty,
                        "graphpack! inputs must have type Input<T>",
                    )
                    .to_compile_error()
                    .into();
                }
            };

            (name, data_type)
        }
        _ => {
            return syn::Error::new_spanned(
                input,
                "graphpack! inputs must have the form |x: Input<T>|",
            )
            .to_compile_error()
            .into();
        }
    };

    let name = name.to_string();

    TokenStream::from(quote! {
        {
            let mut graph = ::tensorflow::Graph::new();
            let mut operation = graph
                .new_operation("Placeholder", #name)
                .expect("failed to create input operation");
            operation
                .set_attr_type("dtype", #data_type)
                .expect("failed to set input data type");
            operation
                .finish()
                .expect("failed to finish input operation");
            graph.graph_def().expect("failed to serialize GraphDef")
        }
    })
}

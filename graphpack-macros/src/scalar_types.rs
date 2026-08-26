use proc_macro2::TokenStream;
use quote::quote;
use syn::{GenericArgument, PathArguments, Type};

#[derive(Clone, Copy)]
pub enum ScalarType {
    F32,
    I32,
    Complex64,
}

impl ScalarType {
    pub fn from_input_type(ty: &Type) -> syn::Result<Self> {
        let Type::Path(type_path) = ty else {
            return Err(syn::Error::new_spanned(ty, "graphpack! closure inputs must have type Input<T>"));
        };
        let segment = type_path.path.segments.last().unwrap();
        if segment.ident != "Input" {
            return Err(syn::Error::new_spanned(ty, "graphpack! closure inputs must have type Input<T>"));
        }
        match &segment.arguments {
            PathArguments::AngleBracketed(arguments) if arguments.args.len() == 1 => {
                match arguments.args.first().unwrap() {
                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("f32") => Ok(Self::F32),
                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("i32") => Ok(Self::I32),
                    GenericArgument::Type(Type::Path(path)) if path.path.segments.last().is_some_and(|s| s.ident == "Complex") => {
                        let complex = path.path.segments.last().unwrap();
                        match &complex.arguments {
                            PathArguments::AngleBracketed(arguments) if arguments.args.len() == 1 => {
                                match arguments.args.first().unwrap() {
                                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("f32") => Ok(Self::Complex64),
                                    other => Err(syn::Error::new_spanned(other, "graphpack! currently supports Input<Complex<f32>>")),
                                }
                            }
                            _ => Err(syn::Error::new_spanned(&complex.arguments, "graphpack! currently supports Input<Complex<f32>>")),
                        }
                    }
                    other => Err(syn::Error::new_spanned(other, "graphpack! currently supports Input<f32>, Input<i32>, and Input<Complex<f32>>")),
                }
            }
            _ => Err(syn::Error::new_spanned(&segment.arguments, "graphpack! inputs must have type Input<T>")),
        }
    }

    pub fn data_type(self) -> TokenStream {
        match self {
            Self::F32 => quote!(::tensorflow::DataType::Float),
            Self::I32 => quote!(::tensorflow::DataType::Int32),
            Self::Complex64 => quote!(::tensorflow::DataType::Complex64),
        }
    }

    pub fn tensor_type(self) -> TokenStream {
        match self {
            Self::F32 => quote!(f32),
            Self::I32 => quote!(i32),
            Self::Complex64 => quote!(::num_complex::Complex<f32>),
        }
    }
}

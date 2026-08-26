use proc_macro2::TokenStream;
use quote::quote;
use syn::{GenericArgument, PathArguments, Type};

#[derive(Clone, Copy)]
pub enum ScalarType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Usize,
    F32,
    F64,
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
                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("bool") => Ok(Self::Bool),
                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("i8") => Ok(Self::I8),
                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("i16") => Ok(Self::I16),
                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("i32") => Ok(Self::I32),
                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("i64") => Ok(Self::I64),
                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("u8") => Ok(Self::U8),
                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("u16") => Ok(Self::U16),
                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("u32") => Ok(Self::U32),
                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("u64") => Ok(Self::U64),
                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("usize") => Ok(Self::Usize),
                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("f32") => Ok(Self::F32),
                    GenericArgument::Type(Type::Path(path)) if path.path.is_ident("f64") => Ok(Self::F64),
                    other => Err(syn::Error::new_spanned(other, "graphpack! unsupported Input<T> scalar type")),
                }
            }
            _ => Err(syn::Error::new_spanned(&segment.arguments, "graphpack! inputs must have type Input<T>")),
        }
    }

    pub fn data_type(self) -> TokenStream {
        match self {
            Self::Bool => quote!(::tensorflow::DataType::Bool),
            Self::I8 => quote!(::tensorflow::DataType::Int8),
            Self::I16 => quote!(::tensorflow::DataType::Int16),
            Self::I32 => quote!(::tensorflow::DataType::Int32),
            Self::I64 => quote!(::tensorflow::DataType::Int64),
            Self::U8 => quote!(::tensorflow::DataType::UInt8),
            Self::U16 => quote!(::tensorflow::DataType::UInt16),
            Self::U32 => quote!(::tensorflow::DataType::UInt32),
            Self::Usize => quote!(::tensorflow::DataType::UInt64),
            Self::U64 => quote!(::tensorflow::DataType::UInt64),
            Self::F32 => quote!(::tensorflow::DataType::Float),
            Self::F64 => quote!(::tensorflow::DataType::Double),
        }
    }

    pub fn tensor_type(self) -> TokenStream {
        match self {
            Self::Bool => quote!(bool),
            Self::I8 => quote!(i8),
            Self::I16 => quote!(i16),
            Self::I32 => quote!(i32),
            Self::I64 => quote!(i64),
            Self::U8 => quote!(u8),
            Self::U16 => quote!(u16),
            Self::U32 => quote!(u32),
            Self::Usize | Self::U64 => quote!(u64),
            Self::F32 => quote!(f32),
            Self::F64 => quote!(f64),
        }
    }
}

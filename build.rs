use std::{env, fs, path::PathBuf, slice};

use prost::Message;

#[derive(Clone, PartialEq, Message)]
struct OpList {
    #[prost(message, repeated, tag = "1")]
    op: Vec<OpDef>,
}

#[derive(Clone, PartialEq, Message)]
struct OpDef {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, repeated, tag = "2")]
    input_arg: Vec<ArgDef>,
    #[prost(message, repeated, tag = "3")]
    output_arg: Vec<ArgDef>,
    #[prost(message, repeated, tag = "4")]
    attr: Vec<AttrDef>,
    #[prost(message, optional, tag = "8")]
    deprecation: Option<OpDeprecation>,
    #[prost(string, tag = "5")]
    summary: String,
    #[prost(string, tag = "6")]
    description: String,
    #[prost(bool, tag = "18")]
    is_commutative: bool,
    #[prost(bool, tag = "16")]
    is_aggregate: bool,
    #[prost(bool, tag = "17")]
    is_stateful: bool,
    #[prost(bool, tag = "19")]
    allows_uninitialized_input: bool,
    #[prost(string, repeated, tag = "20")]
    control_output: Vec<String>,
    #[prost(bool, tag = "21")]
    is_distributed_communication: bool,
}

#[derive(Clone, PartialEq, Message)]
struct ArgDef {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(string, tag = "2")]
    description: String,
    #[prost(int32, tag = "3")]
    r#type: i32,
    #[prost(string, tag = "4")]
    type_attr: String,
    #[prost(string, tag = "5")]
    number_attr: String,
    #[prost(string, tag = "6")]
    type_list_attr: String,
    #[prost(bytes, repeated, tag = "7")]
    handle_data: Vec<Vec<u8>>,
    #[prost(bool, tag = "16")]
    is_ref: bool,
    #[prost(bytes, optional, tag = "17")]
    experimental_full_type: Option<Vec<u8>>,
}

#[derive(Clone, PartialEq, Message)]
struct AttrDef {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(string, tag = "2")]
    r#type: String,
    #[prost(message, optional, tag = "3")]
    default_value: Option<AttrValue>,
    #[prost(string, tag = "4")]
    description: String,
    #[prost(bool, tag = "5")]
    has_minimum: bool,
    #[prost(int64, tag = "6")]
    minimum: i64,
    #[prost(message, optional, tag = "7")]
    allowed_values: Option<AttrValue>,
}

#[derive(Clone, PartialEq, Message)]
struct AttrValue {
    #[prost(oneof = "attr_value::Value", tags = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12")]
    value: Option<attr_value::Value>,
}

mod attr_value {
    #[derive(Clone, PartialEq, prost::Oneof)]
    pub enum Value {
        #[prost(bytes, tag = "1")]
        S(Vec<u8>),
        #[prost(float, tag = "2")]
        F(f32),
        #[prost(int64, tag = "3")]
        I(i64),
        #[prost(bool, tag = "4")]
        B(bool),
        #[prost(message, tag = "5")]
        Type(super::TypeList),
        #[prost(message, tag = "6")]
        Shape(super::Shape),
        #[prost(message, tag = "7")]
        Tensor(super::Tensor),
        #[prost(message, tag = "8")]
        List(super::ListValue),
        #[prost(string, tag = "9")]
        Func(String),
        #[prost(string, tag = "10")]
        Placeholder(String),
        #[prost(message, tag = "11")]
        NameAttrList(super::NameAttrList),
        #[prost(message, tag = "12")]
        FullType(super::FullTypeDef),
    }
}

#[derive(Clone, PartialEq, Message)]
struct TypeList {
    #[prost(int32, repeated, tag = "6")]
    r#type: Vec<i32>,
}

#[derive(Clone, PartialEq, Message)]
struct Shape {
    #[prost(message, repeated, tag = "2")]
    dim: Vec<Dim>,
    #[prost(bool, tag = "3")]
    unknown_rank: bool,
}

#[derive(Clone, PartialEq, Message)]
struct Dim {
    #[prost(int64, tag = "1")]
    size: i64,
    #[prost(string, tag = "2")]
    name: String,
}

#[derive(Clone, PartialEq, Message)]
struct Tensor {
    #[prost(int32, tag = "1")]
    dtype: i32,
    #[prost(message, optional, tag = "2")]
    tensor_shape: Option<Shape>,
}

#[derive(Clone, PartialEq, Message)]
struct ListValue {
    #[prost(bytes, repeated, tag = "2")]
    s: Vec<Vec<u8>>,
    #[prost(float, repeated, tag = "3")]
    f: Vec<f32>,
    #[prost(int64, repeated, tag = "4")]
    i: Vec<i64>,
    #[prost(bool, repeated, tag = "5")]
    b: Vec<bool>,
    #[prost(int32, repeated, tag = "6")]
    r#type: Vec<i32>,
    #[prost(message, repeated, tag = "7")]
    shape: Vec<Shape>,
    #[prost(message, repeated, tag = "8")]
    tensor: Vec<Tensor>,
    #[prost(message, repeated, tag = "9")]
    func: Vec<NameAttrList>,
}

#[derive(Clone, PartialEq, Message)]
struct NameAttrList {
    #[prost(string, tag = "1")]
    name: String,
}

#[derive(Clone, PartialEq, Message)]
struct FullTypeDef {
    #[prost(int32, tag = "1")]
    type_id: i32,
    #[prost(message, repeated, tag = "2")]
    args: Vec<FullTypeDef>,
}

#[derive(Clone, PartialEq, Message)]
struct OpDeprecation {
    #[prost(int32, tag = "1")]
    version: i32,
    #[prost(string, tag = "2")]
    explanation: String,
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let buffer = unsafe { tensorflow_sys::TF_GetAllOpList() };
    if buffer.is_null() {
        panic!("TensorFlow returned a null operation list buffer");
    }

    let bytes = unsafe {
        let buffer_ref = &*buffer;
        if buffer_ref.data.is_null() {
            tensorflow_sys::TF_DeleteBuffer(buffer);
            panic!("TensorFlow returned an operation list with null data");
        }
        slice::from_raw_parts(buffer_ref.data as *const u8, buffer_ref.length)
    };

    let mut op_list = OpList::decode(bytes).unwrap_or_else(|error| {
        unsafe { tensorflow_sys::TF_DeleteBuffer(buffer) };
        panic!("failed to decode TensorFlow operation list: {error}");
    });
    unsafe { tensorflow_sys::TF_DeleteBuffer(buffer) };

    op_list.op.retain(|op| !op.name.starts_with('_'));
    op_list.op.sort_by(|a, b| a.name.cmp(&b.name));
    op_list.op.dedup_by(|a, b| a.name == b.name);

    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"))
        .join("tensorflow_ops.rs");
    let mut generated = String::from(
        "#[derive(Debug, Clone, Copy)]\npub struct TensorFlowOpDef {\n    pub name: &'static str,\n    pub input_args: &'static [TensorFlowArgDef],\n    pub output_args: &'static [TensorFlowArgDef],\n    pub attrs: &'static [TensorFlowAttrDef],\n    pub summary: &'static str,\n    pub description: &'static str,\n    pub is_commutative: bool,\n    pub is_aggregate: bool,\n    pub is_stateful: bool,\n    pub allows_uninitialized_input: bool,\n    pub control_outputs: &'static [&'static str],\n    pub is_distributed_communication: bool,\n}\n\n#[derive(Debug, Clone, Copy)]\npub struct TensorFlowArgDef {\n    pub name: &'static str,\n    pub description: &'static str,\n    pub data_type: i32,\n    pub type_attr: &'static str,\n    pub number_attr: &'static str,\n    pub type_list_attr: &'static str,\n    pub is_ref: bool,\n}\n\n#[derive(Debug, Clone, Copy)]\npub struct TensorFlowAttrDef {\n    pub name: &'static str,\n    pub data_type: &'static str,\n    pub default_value: Option<TensorFlowAttrValue>,\n    pub description: &'static str,\n    pub has_minimum: bool,\n    pub minimum: i64,\n    pub allowed_values: Option<TensorFlowAttrValue>,\n}\n\n#[derive(Debug, Clone, Copy)]\npub enum TensorFlowAttrValue {\n    String(&'static [u8]),\n    Float(f32),\n    Int(i64),\n    Bool(bool),\n    Types(&'static [i32]),\n    Shape(TensorFlowShape),\n    Tensor(TensorFlowTensor),\n    List(TensorFlowListValue),\n    Function(&'static str),\n    Placeholder(&'static str),\n    NameAttrList(&'static str),\n    FullType(TensorFlowFullType),\n}\n\n#[derive(Debug, Clone, Copy)]\npub struct TensorFlowShape { pub dims: &'static [TensorFlowDim], pub unknown_rank: bool }\n#[derive(Debug, Clone, Copy)]\npub struct TensorFlowDim { pub size: i64, pub name: &'static str }\n#[derive(Debug, Clone, Copy)]\npub struct TensorFlowTensor { pub data_type: i32, pub shape: Option<TensorFlowShape> }\n#[derive(Debug, Clone, Copy)]\npub struct TensorFlowListValue {\n    pub strings: &'static [&'static [u8]], pub floats: &'static [f32], pub ints: &'static [i64],\n    pub bools: &'static [bool], pub types: &'static [i32], pub shapes: &'static [TensorFlowShape],\n    pub tensors: &'static [TensorFlowTensor], pub functions: &'static [&'static str],\n}\n#[derive(Debug, Clone, Copy)]\npub struct TensorFlowFullType { pub type_id: i32, pub args: &'static [TensorFlowFullType] }\n\n",
    );

    for (index, op) in op_list.op.iter().enumerate() {
        generated.push_str(&format!("static INPUTS_{index}: &[TensorFlowArgDef] = &[\n"));
        for arg in &op.input_arg {
            generated.push_str(&format!(
                "    TensorFlowArgDef {{ name: {:?}, description: {:?}, data_type: {}, type_attr: {:?}, number_attr: {:?}, type_list_attr: {:?}, is_ref: {} }},\n",
                arg.name, arg.description, arg.r#type, arg.type_attr, arg.number_attr, arg.type_list_attr, arg.is_ref
            ));
        }
        generated.push_str("];\n");

        generated.push_str(&format!("static OUTPUTS_{index}: &[TensorFlowArgDef] = &[\n"));
        for arg in &op.output_arg {
            generated.push_str(&format!(
                "    TensorFlowArgDef {{ name: {:?}, description: {:?}, data_type: {}, type_attr: {:?}, number_attr: {:?}, type_list_attr: {:?}, is_ref: {} }},\n",
                arg.name, arg.description, arg.r#type, arg.type_attr, arg.number_attr, arg.type_list_attr, arg.is_ref
            ));
        }
        generated.push_str("];\n");

        generated.push_str(&format!("static ATTRS_{index}: &[TensorFlowAttrDef] = &[\n"));
        for attr in &op.attr {
            generated.push_str(&format!(
                "    TensorFlowAttrDef {{ name: {:?}, data_type: {:?}, default_value: None, description: {:?}, has_minimum: {}, minimum: {}, allowed_values: None }},\n",
                attr.name, attr.r#type, attr.description, attr.has_minimum, attr.minimum
            ));
        }
        generated.push_str("];\n");
    }

    generated.push_str("\npub static TENSORFLOW_OPS: &[TensorFlowOpDef] = &[\n");
    for (index, op) in op_list.op.iter().enumerate() {
        generated.push_str(&format!(
            "    TensorFlowOpDef {{ name: {:?}, input_args: INPUTS_{index}, output_args: OUTPUTS_{index}, attrs: ATTRS_{index}, summary: {:?}, description: {:?}, is_commutative: {}, is_aggregate: {}, is_stateful: {}, allows_uninitialized_input: {}, control_outputs: &{}, is_distributed_communication: {} }},\n",
            op.name, op.summary, op.description, op.is_commutative, op.is_aggregate,
            op.is_stateful, op.allows_uninitialized_input,
            format!("[{}]", op.control_output.iter().map(|v| format!("{v:?}")).collect::<Vec<_>>().join(", ")),
            op.is_distributed_communication
        ));
    }
    generated.push_str("];\n");

    fs::write(output, generated).expect("failed to write generated TensorFlow operation metadata");
}

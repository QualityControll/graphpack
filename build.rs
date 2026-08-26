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
    #[prost(bytes, optional, tag = "3")]
    default_value: Option<Vec<u8>>,
    #[prost(string, tag = "4")]
    description: String,
    #[prost(bool, tag = "5")]
    has_minimum: bool,
    #[prost(int64, tag = "6")]
    minimum: i64,
    #[prost(bytes, optional, tag = "7")]
    allowed_values: Option<Vec<u8>>,
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

    println!(
        "cargo:warning=TensorFlow exposes {} public operations",
        op_list.op.len()
    );

    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"))
        .join("tensorflow_ops.rs");
    let mut generated = String::from(
        "#[derive(Debug, Clone, Copy)]\npub struct TensorFlowOpDef {\n    pub name: &'static str,\n    pub serialized: &'static [u8],\n}\n\n",
    );

    generated.push_str("pub static TENSORFLOW_OPS: &[TensorFlowOpDef] = &[\n");
    for op in &op_list.op {
        let serialized = op.encode_to_vec();
        generated.push_str(&format!(
            "    TensorFlowOpDef {{ name: {:?}, serialized: &{:?} }},\n",
            op.name, serialized
        ));
    }
    generated.push_str("];\n");

    fs::write(output, generated).expect("failed to write generated TensorFlow operation metadata");
}

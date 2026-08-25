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

fn rust_string(value: &str) -> String {
    format!("{value:?}")
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

    let op_list = OpList::decode(bytes).unwrap_or_else(|error| {
        unsafe { tensorflow_sys::TF_DeleteBuffer(buffer) };
        panic!("failed to decode TensorFlow operation list: {error}");
    });

    unsafe { tensorflow_sys::TF_DeleteBuffer(buffer) };

    let mut ops: Vec<OpDef> = op_list
        .op
        .into_iter()
        .filter(|op| !op.name.starts_with('_'))
        .collect();
    ops.sort_by(|a, b| a.name.cmp(&b.name));
    ops.dedup_by(|a, b| a.name == b.name);

    println!(
        "cargo:warning=TensorFlow exposes {} public operations",
        ops.len()
    );

    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"))
        .join("tensorflow_ops.rs");
    let mut generated = String::from(
        "#[derive(Debug, Clone, Copy)]\npub struct TensorFlowOpDef {\n    pub name: &'static str,\n    pub input_args: &'static [TensorFlowArgDef],\n    pub output_args: &'static [TensorFlowArgDef],\n    pub attrs: &'static [TensorFlowAttrDef],\n    pub deprecation: Option<TensorFlowOpDeprecation>,\n    pub summary: &'static str,\n    pub description: &'static str,\n    pub is_commutative: bool,\n    pub is_aggregate: bool,\n    pub is_stateful: bool,\n    pub allows_uninitialized_input: bool,\n    pub control_outputs: &'static [&'static str],\n    pub is_distributed_communication: bool,\n}\n\n#[derive(Debug, Clone, Copy)]\npub struct TensorFlowArgDef {\n    pub name: &'static str,\n    pub description: &'static str,\n    pub data_type: i32,\n    pub type_attr: &'static str,\n    pub number_attr: &'static str,\n    pub type_list_attr: &'static str,\n    pub handle_data: &'static [&'static [u8]],\n    pub is_ref: bool,\n    pub experimental_full_type: Option<&'static [u8]>,\n}\n\n#[derive(Debug, Clone, Copy)]\npub struct TensorFlowAttrDef {\n    pub name: &'static str,\n    pub data_type: &'static str,\n    pub default_value: Option<&'static [u8]>,\n    pub description: &'static str,\n    pub has_minimum: bool,\n    pub minimum: i64,\n    pub allowed_values: Option<&'static [u8]>,\n}\n\n#[derive(Debug, Clone, Copy)]\npub struct TensorFlowOpDeprecation {\n    pub version: i32,\n    pub explanation: &'static str,\n}\n\n",
    );

    for (index, op) in ops.iter().enumerate() {
        generated.push_str(&format!("static INPUT_ARGS_{index}: &[TensorFlowArgDef] = &[\n"));
        for arg in &op.input_arg {
            generated.push_str(&format!(
                "    TensorFlowArgDef {{ name: {}, description: {}, data_type: {}, type_attr: {}, number_attr: {}, type_list_attr: {}, handle_data: &[], is_ref: {}, experimental_full_type: {} }},\n",
                rust_string(&arg.name), rust_string(&arg.description), arg.r#type,
                rust_string(&arg.type_attr), rust_string(&arg.number_attr),
                rust_string(&arg.type_list_attr), arg.is_ref,
                arg.experimental_full_type.as_deref().map(|v| format!("Some(&{v:?})")).unwrap_or_else(|| "None"),
            ));
        }
        generated.push_str("];\n");

        generated.push_str(&format!("static OUTPUT_ARGS_{index}: &[TensorFlowArgDef] = &[\n"));
        for arg in &op.output_arg {
            generated.push_str(&format!(
                "    TensorFlowArgDef {{ name: {}, description: {}, data_type: {}, type_attr: {}, number_attr: {}, type_list_attr: {}, handle_data: &[], is_ref: {}, experimental_full_type: {} }},\n",
                rust_string(&arg.name), rust_string(&arg.description), arg.r#type,
                rust_string(&arg.type_attr), rust_string(&arg.number_attr),
                rust_string(&arg.type_list_attr), arg.is_ref,
                arg.experimental_full_type.as_deref().map(|v| format!("Some(&{v:?})")).unwrap_or_else(|| "None"),
            ));
        }
        generated.push_str("];\n");

        generated.push_str(&format!("static ATTRS_{index}: &[TensorFlowAttrDef] = &[\n"));
        for attr in &op.attr {
            generated.push_str(&format!(
                "    TensorFlowAttrDef {{ name: {}, data_type: {}, default_value: {}, description: {}, has_minimum: {}, minimum: {}, allowed_values: {} }},\n",
                rust_string(&attr.name), rust_string(&attr.r#type),
                attr.default_value.as_deref().map(|v| format!("Some(&{v:?})")).unwrap_or_else(|| "None"),
                rust_string(&attr.description), attr.has_minimum, attr.minimum,
                attr.allowed_values.as_deref().map(|v| format!("Some(&{v:?})")).unwrap_or_else(|| "None"),
            ));
        }
        generated.push_str("];\n");
    }

    generated.push_str("\npub static TENSORFLOW_OPS: &[TensorFlowOpDef] = &[\n");
    for (index, op) in ops.iter().enumerate() {
        let deprecation = op.deprecation.as_ref().map(|d| {
            format!(
                "Some(TensorFlowOpDeprecation {{ version: {}, explanation: {} }})",
                d.version,
                rust_string(&d.explanation)
            )
        }).unwrap_or_else(|| "None".to_string());
        generated.push_str(&format!(
            "    TensorFlowOpDef {{ name: {}, input_args: INPUT_ARGS_{index}, output_args: OUTPUT_ARGS_{index}, attrs: ATTRS_{index}, deprecation: {deprecation}, summary: {}, description: {}, is_commutative: {}, is_aggregate: {}, is_stateful: {}, allows_uninitialized_input: {}, control_outputs: &{}, is_distributed_communication: {} }},\n",
            rust_string(&op.name), rust_string(&op.summary), rust_string(&op.description),
            op.is_commutative, op.is_aggregate, op.is_stateful,
            op.allows_uninitialized_input,
            format!("[{}]", op.control_output.iter().map(|v| rust_string(v)).collect::<Vec<_>>().join(", ")),
            op.is_distributed_communication,
        ));
    }
    generated.push_str("];\n");

    fs::write(output, generated).expect("failed to write generated TensorFlow operation metadata");
}

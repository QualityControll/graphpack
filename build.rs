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

    let mut names: Vec<String> = op_list.op.into_iter().map(|op| op.name).collect();
    names.sort();
    names.dedup();

    println!("cargo:warning=TensorFlow exposes {} operations", names.len());

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    let output = out_dir.join("tensorflow_ops.rs");
    let mut generated = String::from("pub const TENSORFLOW_OPS: &[&str] = &[\n");
    for name in &names {
        generated.push_str(&format!("    {name:?},\n"));
    }
    generated.push_str("];\n");

    fs::write(&output, generated).expect("failed to write generated TensorFlow operation list");
}

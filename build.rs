use std::{env, fs, path::PathBuf, slice};

use prost07::Message;
use tensorflow_proto::tensorflow::core::framework::op_def::OpList;

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

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    let output = out_dir.join("tensorflow_ops.rs");

    // Keep the complete, official TensorFlow protobuf representation. The
    // generated API exposes typed OpDef values rather than flattening or
    // reimplementing TensorFlow's protobuf schema in GraphPack.
    let mut generated = String::from(
        "use tensorflow_proto::tensorflow::core::framework::op_def::OpDef;\n\n"
    );
    generated.push_str("pub fn tensorflow_ops() -> Vec<OpDef> {\n    vec![\n");

    for op in &op_list.op {
        let bytes = op.encode_to_vec();
        generated.push_str(&format!(
            "        OpDef::decode(&{:?}[..]).expect(\"generated TensorFlow OpDef failed to decode\"),\n",
            bytes
        ));
    }

    generated.push_str("    ]\n}\n");

    fs::write(output, generated)
        .expect("failed to write generated TensorFlow operation metadata");
}

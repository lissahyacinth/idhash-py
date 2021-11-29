use std::{error::Error, sync::Arc};

use arrow::{
    array::ArrayRef,
    datatypes::{DataType, Field, Schema, TimeUnit},
    ffi,
    record_batch::RecordBatch,
};
use idhash::{calculate_idhash, config::IdHashConfigBuilder};
use pyo3::{ffi::Py_uintptr_t, prelude::*};

#[pyclass]
struct IDHasher {
    hash: u128,
    schema: Arc<Schema>,
    field_names: Vec<String>,
}

#[pymethods]
impl IDHasher {
    #[new]
    fn new(field_names: Vec<String>, field_types: Vec<String>) -> Self {
        IDHasher {
            hash: 0,
            schema: get_schema(field_names.clone(), field_types),
            field_names,
        }
    }

    /// Write an entire Arrow Table as Batches to Hasher
    fn write_batches(&mut self, batches: Vec<&PyAny>, delta: String) {
        let config = IdHashConfigBuilder::new().build();
        let hash: u128 = calculate_idhash(
            from_py_record_batches(&batches, self.schema.clone(), self.field_names.len()),
            self.schema.clone(),
            config,
        );
        match delta.as_ref() {
            "Add" => *self += hash,
            "Remove" => *self -= hash,
            _ => unimplemented!(),
        }
    }

    fn finalize(&self) -> u128 {
        self.hash
    }
}

impl std::ops::AddAssign<u128> for IDHasher {
    fn add_assign(&mut self, other: u128) {
        self.hash = self.hash.wrapping_add(other);
    }
}

impl std::ops::SubAssign<u128> for IDHasher {
    fn sub_assign(&mut self, other: u128) {
        self.hash = self.hash.wrapping_sub(other);
    }
}

/// Convert Python Record Batch to a Rust Record Batch
///
/// Method uses the C FFI layer and shared syntax between the Rust Arrow and C Arrow implementations.
///
/// Method partially adopted from (https://github.com/pola-rs/polars/blob/629f5012bcefaa3c9a9c1a236e64dc057e8d472c/py-polars/src/arrow_interop/to_rust.rs#L32-L66)
/// with alterations to use Arrow-rs instead of Arrow2, as there is minimal concern WRT transmutation.
pub fn array_to_rust(obj: &PyAny) -> Result<ArrayRef, Box<dyn Error>> {
    let array = Box::new(ffi::Ffi_ArrowArray::empty());
    let schema = Box::new(ffi::Ffi_ArrowSchema::empty());

    let array_ptr = &*array as *const ffi::Ffi_ArrowArray;
    let schema_ptr = &*schema as *const ffi::Ffi_ArrowSchema;

    obj.call_method1(
        "_export_to_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;
    unsafe {
        let field = ffi::import_field_from_c(schema.as_ref())?;
        let array = ffi::import_array_from_c(array, &field)?;
        Ok(array.into())
    }
}

fn convert_to_data_type(data_type: &String) -> DataType {
    match data_type.as_ref() {
        "int16" => DataType::Int16,
        "int32" => DataType::Int32,
        "int64" => DataType::Int64,
        "float16" => DataType::Float16,
        "float32" => DataType::Float32,
        "float64" => DataType::Float64,
        "string" => DataType::Utf8,
        "bool" => DataType::Boolean,
        "datetime64[ns]" => DataType::Timestamp(TimeUnit::Nanosecond, Some("".to_string())),
        "object" => DataType::Utf8,
        _ => panic!("Unknown Data Type {}", data_type),
    }
}

fn get_schema(field_names: Vec<String>, field_types: Vec<String>) -> Arc<Schema> {
    Arc::new(Schema::new(
        field_names
            .iter()
            .zip(field_types.iter().map(convert_to_data_type))
            .map(|(name, data_type)| Field::new(name, data_type, false))
            .collect(),
    ))
}

fn from_py_record_batches<'a>(
    record_batches: &'a [&PyAny],
    schema: Arc<Schema>,
    total_fields: usize,
) -> Box<dyn Iterator<Item = RecordBatch> + 'a> {
    let batch_iter = record_batches.iter().map(move |rb| {
        let columns = (0..total_fields)
            .map(|i| {
                let array = rb.call_method1("column", (i,)).unwrap();
                array_to_rust(array).unwrap()
            })
            .collect::<Vec<ArrayRef>>();
        RecordBatch::try_new(schema.clone(), columns).unwrap()
    });
    Box::new(batch_iter)
}

/// Calculate the IDHash for a set of Record Batches
///
/// # Arguments
/// * `record_batches` - A list of record batches
/// * `field_names` - The fields contained in each record batch
/// * `field_types` - The type associated with each field
///
/// An identical number of field_names and field_types must be passed
#[pyfunction]
pub fn id_hash(
    record_batches: Vec<&PyAny>,
    field_names: Vec<String>,
    field_types: Vec<String>,
) -> PyResult<u128> {
    assert_eq!(field_names.len(), field_types.len());
    let total_fields = field_names.len();
    let schema = get_schema(field_names, field_types);
    let config = IdHashConfigBuilder::new().build();
    Ok(calculate_idhash(
        from_py_record_batches(&record_batches, schema.clone(), total_fields),
        schema,
        config,
    ))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn idhash(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(id_hash, m)?)?;
    m.add_class::<IDHasher>()?;
    Ok(())
}

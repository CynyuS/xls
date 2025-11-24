fn hello_xls(hello_string: u8[11]) {
    // Trace the value to stderr as output.
    trace!(hello_string);
}

#[test]
fn hello_test() {
  hello_xls("Hello, XLS!")
}
file(REMOVE_RECURSE
  "../lib/libtvm_ffi.pdb"
  "../lib/libtvm_ffi.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/tvm_ffi_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

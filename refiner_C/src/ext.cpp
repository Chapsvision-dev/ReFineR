#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

// #include <vector>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace refiner_C {

  // Defines the operators
  TORCH_LIBRARY(refiner_C, m) {
    m.def("hist_conv(Tensor A, Tensor weights, Tensor bins, int kernel_size, float sigma) -> Tensor");
    m.def("hist_conv_backward(Tensor grad_output, Tensor A, Tensor weights, Tensor bins, int kernel_size, float sigma) -> (Tensor, Tensor)");

    m.def("sample_conv(Tensor A, int kernel_size, int stride) -> Tensor");
    m.def("sample_conv_backward(Tensor grad_A, Tensor A, int kernel_size, int stride) -> Tensor");

    m.def("target_detect(Tensor A, int radius, int radiusin, int min_color_std, float max_offset, float max_line_cov, float min_angle, float cross_thresh, float thresh) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

    m.def("rasterize(Tensor pc, Tensor chans, Tensor extr, Tensor intr, int W, int H, float dist) -> (Tensor, Tensor, Tensor)");

  }

}
#include <tuple>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

using namespace torch::indexing;

namespace refiner_C {
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rasterize(
        const torch::Tensor& xyz,
        const torch::Tensor& chans,
        const torch::Tensor& extr,
        const torch::Tensor& intr,
        int64_t W, int64_t H,
        const double dist = 0.
    ){  
        TORCH_INTERNAL_ASSERT(xyz.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(extr.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(intr.device().type() == at::DeviceType::CPU);

        TORCH_CHECK(xyz.dtype() == torch::kFloat32);
        TORCH_CHECK(extr.dtype() == torch::kFloat32);
        TORCH_CHECK(intr.dtype() == torch::kFloat32);

        TORCH_CHECK(extr.sizes().equals({4, 4}));
        TORCH_CHECK(intr.sizes().equals({3, 3}));

        // Project
        torch::Tensor points = torch::transpose(xyz, 0, 1);
        if (dist == 0.){
            torch::Tensor persp = torch::matmul(intr, extr.index({Slice(0, 3), Slice()}));
            points = torch::matmul(
                persp.index({Slice(0, 3), Slice(0, 3)}), points
            );
            points = points + persp.index({Slice(0, 3), Slice(3, 4)});
            points.index_put_({Slice(0, 2)}, points.index({Slice(0, 2)}) / points.index({Slice(2, 3)}));
            
        }
        else{
            points = torch::matmul(
                extr.index({Slice(0, 3), Slice(0, 3)}), points
            );
            points = points + extr.index({Slice(0, 3), Slice(3, 4)});
            points.index_put_({Slice(0, 2)}, points.index({Slice(0, 2)}) / points.index({Slice(2, 3)}));

            torch::Tensor dists = points.index({0}) * points.index({0}) + points.index({1}) * points.index({1});

            dists = 2 / (1 + torch::sqrt(1 - 4 * dist * dists));
            points.index_put_({Slice(0, 2)}, points.index({Slice(0, 2)}) * dists);
            points.index_put_({Slice(0, 2)},
                torch::matmul(
                        intr.index({Slice(0, 2), Slice(0, 2)}), points.index({Slice(0, 2)})
                    )
            );

            points.index_put_({Slice(0, 2)},
                points.index({Slice(0, 2)}) + intr.index({Slice(0, 2), Slice(2, 3)})
            );
        }

        // Frustrum filter
        torch::Tensor mask = points.index({2}) > 1;
        mask = mask * (points.index({0}) < W);
        mask = mask * (points.index({0}) > 0);
        mask = mask * (points.index({1}) < H);
        mask = mask * (points.index({1}) > 0);
        points = points.index({Slice(), mask});
        torch::Tensor channels = chans.index({mask});
        
        // Rasterize
        const int N = points.sizes()[1];
        const int C = channels.sizes()[1];

        auto depth_options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
        torch::Tensor depth = torch::full({H, W}, 0., depth_options);

        auto pixel_coords_options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
        torch::Tensor pixel_coords = torch::full({H, W, 2}, 0., pixel_coords_options);

        auto img_options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
        torch::Tensor img = torch::full({H, W, C}, 0, img_options);

        auto points_acc = points.accessor<float, 2>();
        auto pixel_coords_acc = pixel_coords.accessor<float, 3>();
        auto depth_acc = depth.accessor<float, 2>();
        auto img_acc = img.accessor<float, 3>();
        auto chan_acc = channels.accessor<float, 2>();

        float xf, yf;
        int x, y;
        float z;
        for (int i = 0; i< N; i++){
            xf = points_acc[0][i];
            yf = points_acc[1][i];
            x = (int) xf;
            y = (int) yf;
            z = points_acc[2][i];
            if (depth_acc[y][x] > z || depth_acc[y][x] == 0) {
                depth_acc[y][x] = z;
                for (int j = 0; j < C; j++){
                    img_acc[y][x][j] = chan_acc[i][j];
                }

                pixel_coords_acc[y][x][0] = xf;
                pixel_coords_acc[y][x][1] = yf;
            }
        }
        
        return std::make_tuple(img, depth, pixel_coords);
    } 

    TORCH_LIBRARY_IMPL(refiner_C, CPU, m) {
        m.impl("rasterize", &rasterize);
    } 
}
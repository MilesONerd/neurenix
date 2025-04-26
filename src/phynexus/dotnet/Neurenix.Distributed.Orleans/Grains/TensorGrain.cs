using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Neurenix.Distributed.Orleans.Interfaces;
using Orleans;

namespace Neurenix.Distributed.Orleans.Grains
{
    public class TensorGrain : Grain, ITensorGrain
    {
        private readonly ILogger<TensorGrain> _logger;
        private float[]? _data;
        private int[]? _shape;

        public TensorGrain(ILogger<TensorGrain> logger)
        {
            _logger = logger;
        }

        public Task<float[]> GetData()
        {
            return Task.FromResult(_data ?? Array.Empty<float>());
        }

        public Task SetData(float[] data)
        {
            _data = data;
            return Task.CompletedTask;
        }

        public Task<float[]> Add(float[] other)
        {
            if (_data == null || other == null || _data.Length != other.Length)
            {
                throw new ArgumentException("Tensors must have the same shape for addition");
            }

            var result = new float[_data.Length];
            for (int i = 0; i < _data.Length; i++)
            {
                result[i] = _data[i] + other[i];
            }

            return Task.FromResult(result);
        }

        public Task<float[]> Multiply(float[] other)
        {
            if (_data == null || other == null || _data.Length != other.Length)
            {
                throw new ArgumentException("Tensors must have the same shape for element-wise multiplication");
            }

            var result = new float[_data.Length];
            for (int i = 0; i < _data.Length; i++)
            {
                result[i] = _data[i] * other[i];
            }

            return Task.FromResult(result);
        }

        public Task<float[]> MatMul(float[] other, int[] otherShape)
        {
            if (_data == null || _shape == null || otherShape == null || _shape.Length != 2 || otherShape.Length != 2)
            {
                throw new ArgumentException("Matrix multiplication requires 2D tensors with data");
            }

            if (_shape[1] != otherShape[0])
            {
                throw new ArgumentException($"Incompatible shapes for matrix multiplication: ({_shape[0]},{_shape[1]}) and ({otherShape[0]},{otherShape[1]})");
            }

            int m = _shape[0];
            int n = _shape[1];
            int p = otherShape[1];

            var result = new float[m * p];

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < n; k++)
                    {
                        sum += _data[i * n + k] * other[k * p + j];
                    }
                    result[i * p + j] = sum;
                }
            }

            return Task.FromResult(result);
        }

        public Task<int[]> GetShape()
        {
            return Task.FromResult(_shape ?? Array.Empty<int>());
        }

        public Task SetShape(int[] shape)
        {
            _shape = shape;
            return Task.CompletedTask;
        }
    }
}

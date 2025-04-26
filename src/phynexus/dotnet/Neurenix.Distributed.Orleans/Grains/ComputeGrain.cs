using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Neurenix.Distributed.Orleans.Interfaces;
using Orleans;

namespace Neurenix.Distributed.Orleans.Grains
{
    public class ComputeGrain : Grain, IComputeGrain
    {
        private readonly ILogger<ComputeGrain> _logger;
        private readonly Dictionary<string, float[]> _modelParameters = new();

        public ComputeGrain(ILogger<ComputeGrain> logger)
        {
            _logger = logger;
        }

        public Task<float[]> Forward(string modelId, float[] input)
        {
            if (!_modelParameters.TryGetValue(modelId, out var parameters))
            {
                throw new ArgumentException($"Model {modelId} not found");
            }

            var output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] * (parameters.Length > i ? parameters[i] : 1.0f);
            }

            return Task.FromResult(output);
        }

        public Task<float[]> Backward(string modelId, float[] gradients)
        {
            if (!_modelParameters.TryGetValue(modelId, out var parameters))
            {
                throw new ArgumentException($"Model {modelId} not found");
            }

            var paramGradients = new float[parameters.Length];
            for (int i = 0; i < paramGradients.Length; i++)
            {
                paramGradients[i] = gradients.Length > i ? gradients[i] : 0.0f;
            }

            return Task.FromResult(paramGradients);
        }

        public Task<bool> UpdateModel(string modelId, float[] parameters)
        {
            _modelParameters[modelId] = parameters;
            return Task.FromResult(true);
        }

        public Task<float[]> GetModelParameters(string modelId)
        {
            if (!_modelParameters.TryGetValue(modelId, out var parameters))
            {
                throw new ArgumentException($"Model {modelId} not found");
            }

            return Task.FromResult(parameters);
        }
    }
}

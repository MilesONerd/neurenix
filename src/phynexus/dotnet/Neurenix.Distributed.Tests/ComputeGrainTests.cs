using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using Neurenix.Distributed.Orleans.Grains;
using Xunit;

namespace Neurenix.Distributed.Tests
{
    public class ComputeGrainTests
    {
        private readonly Mock<ILogger<ComputeGrain>> _loggerMock;
        private readonly ComputeGrain _computeGrain;

        public ComputeGrainTests()
        {
            _loggerMock = new Mock<ILogger<ComputeGrain>>();
            _computeGrain = new ComputeGrain(_loggerMock.Object);
        }

        [Fact]
        public async Task UpdateModel_ShouldStoreParameters()
        {
            var modelId = "test-model";
            var parameters = new float[] { 1.0f, 2.0f, 3.0f };

            var result = await _computeGrain.UpdateModel(modelId, parameters);

            Assert.True(result);
        }

        [Fact]
        public async Task GetModelParameters_ShouldReturnStoredParameters()
        {
            var modelId = "test-model";
            var parameters = new float[] { 1.0f, 2.0f, 3.0f };
            await _computeGrain.UpdateModel(modelId, parameters);

            var result = await _computeGrain.GetModelParameters(modelId);

            Assert.Equal(parameters, result);
        }

        [Fact]
        public async Task GetModelParameters_WithNonExistentModel_ShouldThrowException()
        {
            var modelId = "non-existent-model";

            await Assert.ThrowsAsync<ArgumentException>(() => _computeGrain.GetModelParameters(modelId));
        }

        [Fact]
        public async Task Forward_ShouldReturnTransformedInput()
        {
            var modelId = "test-model";
            var parameters = new float[] { 2.0f, 3.0f, 4.0f };
            var input = new float[] { 1.0f, 2.0f, 3.0f };
            var expected = new float[] { 2.0f, 6.0f, 12.0f };

            await _computeGrain.UpdateModel(modelId, parameters);

            var result = await _computeGrain.Forward(modelId, input);

            Assert.Equal(expected, result);
        }

        [Fact]
        public async Task Forward_WithNonExistentModel_ShouldThrowException()
        {
            var modelId = "non-existent-model";
            var input = new float[] { 1.0f, 2.0f, 3.0f };

            await Assert.ThrowsAsync<ArgumentException>(() => _computeGrain.Forward(modelId, input));
        }

        [Fact]
        public async Task Backward_ShouldReturnGradients()
        {
            var modelId = "test-model";
            var parameters = new float[] { 2.0f, 3.0f, 4.0f };
            var gradients = new float[] { 0.1f, 0.2f, 0.3f };

            await _computeGrain.UpdateModel(modelId, parameters);

            var result = await _computeGrain.Backward(modelId, gradients);

            Assert.Equal(gradients, result);
        }

        [Fact]
        public async Task Backward_WithNonExistentModel_ShouldThrowException()
        {
            var modelId = "non-existent-model";
            var gradients = new float[] { 0.1f, 0.2f, 0.3f };

            await Assert.ThrowsAsync<ArgumentException>(() => _computeGrain.Backward(modelId, gradients));
        }
    }
}

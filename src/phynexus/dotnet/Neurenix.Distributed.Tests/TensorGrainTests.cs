using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using Neurenix.Distributed.Orleans.Grains;
using Xunit;

namespace Neurenix.Distributed.Tests
{
    public class TensorGrainTests
    {
        private readonly Mock<ILogger<TensorGrain>> _loggerMock;
        private readonly TensorGrain _tensorGrain;

        public TensorGrainTests()
        {
            _loggerMock = new Mock<ILogger<TensorGrain>>();
            _tensorGrain = new TensorGrain(_loggerMock.Object);
        }

        [Fact]
        public async Task SetAndGetData_ShouldWorkCorrectly()
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };

            await _tensorGrain.SetData(data);
            var result = await _tensorGrain.GetData();

            Assert.Equal(data, result);
        }

        [Fact]
        public async Task SetAndGetShape_ShouldWorkCorrectly()
        {
            var shape = new int[] { 2, 2 };

            await _tensorGrain.SetShape(shape);
            var result = await _tensorGrain.GetShape();

            Assert.Equal(shape, result);
        }

        [Fact]
        public async Task Add_ShouldReturnCorrectResult()
        {
            var data1 = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var data2 = new float[] { 5.0f, 6.0f, 7.0f, 8.0f };
            var expected = new float[] { 6.0f, 8.0f, 10.0f, 12.0f };

            await _tensorGrain.SetData(data1);

            var result = await _tensorGrain.Add(data2);

            Assert.Equal(expected, result);
        }

        [Fact]
        public async Task Multiply_ShouldReturnCorrectResult()
        {
            var data1 = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var data2 = new float[] { 5.0f, 6.0f, 7.0f, 8.0f };
            var expected = new float[] { 5.0f, 12.0f, 21.0f, 32.0f };

            await _tensorGrain.SetData(data1);

            var result = await _tensorGrain.Multiply(data2);

            Assert.Equal(expected, result);
        }

        [Fact]
        public async Task MatMul_ShouldReturnCorrectResult()
        {
            var data1 = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape1 = new int[] { 2, 2 };
            var data2 = new float[] { 5.0f, 6.0f, 7.0f, 8.0f };
            var shape2 = new int[] { 2, 2 };
            var expected = new float[] { 19.0f, 22.0f, 43.0f, 50.0f };

            await _tensorGrain.SetData(data1);
            await _tensorGrain.SetShape(shape1);

            var result = await _tensorGrain.MatMul(data2, shape2);

            Assert.Equal(expected, result);
        }

        [Fact]
        public async Task Add_WithIncompatibleShapes_ShouldThrowException()
        {
            var data1 = new float[] { 1.0f, 2.0f, 3.0f };
            var data2 = new float[] { 4.0f, 5.0f };

            await _tensorGrain.SetData(data1);

            await Assert.ThrowsAsync<ArgumentException>(() => _tensorGrain.Add(data2));
        }

        [Fact]
        public async Task MatMul_WithIncompatibleShapes_ShouldThrowException()
        {
            var data1 = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var shape1 = new int[] { 2, 2 };
            var data2 = new float[] { 5.0f, 6.0f, 7.0f };
            var shape2 = new int[] { 3, 1 };

            await _tensorGrain.SetData(data1);
            await _tensorGrain.SetShape(shape1);

            await Assert.ThrowsAsync<ArgumentException>(() => _tensorGrain.MatMul(data2, shape2));
        }
    }
}

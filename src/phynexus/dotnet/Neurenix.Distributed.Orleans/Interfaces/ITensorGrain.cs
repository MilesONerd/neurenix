using System.Threading.Tasks;
using Orleans;

namespace Neurenix.Distributed.Orleans.Interfaces
{
    public interface ITensorGrain : IGrainWithStringKey
    {
        Task<float[]> GetData();
        Task SetData(float[] data);
        Task<float[]> Add(float[] other);
        Task<float[]> Multiply(float[] other);
        Task<float[]> MatMul(float[] other, int[] otherShape);
        Task<int[]> GetShape();
        Task SetShape(int[] shape);
    }
}

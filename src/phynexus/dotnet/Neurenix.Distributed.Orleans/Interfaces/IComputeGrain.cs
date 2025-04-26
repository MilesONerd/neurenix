using System.Threading.Tasks;
using Orleans;

namespace Neurenix.Distributed.Orleans.Interfaces
{
    public interface IComputeGrain : IGrainWithStringKey
    {
        Task<float[]> Forward(string modelId, float[] input);
        Task<float[]> Backward(string modelId, float[] gradients);
        Task<bool> UpdateModel(string modelId, float[] parameters);
        Task<float[]> GetModelParameters(string modelId);
    }
}

using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Orleans;
using Neurenix.Distributed.Orleans.Interfaces;

namespace Neurenix.Distributed.Api.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class TensorController : ControllerBase
    {
        private readonly ILogger<TensorController> _logger;
        private readonly IClusterClient _clusterClient;

        public TensorController(ILogger<TensorController> logger, IClusterClient clusterClient)
        {
            _logger = logger;
            _clusterClient = clusterClient;
        }

        [HttpGet("{tensorId}")]
        public async Task<IActionResult> GetTensor(string tensorId)
        {
            try
            {
                var grain = _clusterClient.GetGrain<ITensorGrain>(tensorId);
                var data = await grain.GetData();
                var shape = await grain.GetShape();

                return Ok(new { data, shape });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting tensor {TensorId}", tensorId);
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpPost("{tensorId}")]
        public async Task<IActionResult> CreateTensor(string tensorId, [FromBody] TensorData tensorData)
        {
            try
            {
                if (tensorData.Data == null || tensorData.Shape == null)
                {
                    return BadRequest("Data and Shape are required");
                }

                var grain = _clusterClient.GetGrain<ITensorGrain>(tensorId);
                await grain.SetData(tensorData.Data);
                await grain.SetShape(tensorData.Shape);

                return Ok();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating tensor {TensorId}", tensorId);
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpPost("{tensorId}/add")]
        public async Task<IActionResult> AddTensors(string tensorId, [FromBody] float[] otherData)
        {
            try
            {
                var grain = _clusterClient.GetGrain<ITensorGrain>(tensorId);
                var result = await grain.Add(otherData);

                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error adding tensors for {TensorId}", tensorId);
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpPost("{tensorId}/multiply")]
        public async Task<IActionResult> MultiplyTensors(string tensorId, [FromBody] float[] otherData)
        {
            try
            {
                var grain = _clusterClient.GetGrain<ITensorGrain>(tensorId);
                var result = await grain.Multiply(otherData);

                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error multiplying tensors for {TensorId}", tensorId);
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpPost("{tensorId}/matmul")]
        public async Task<IActionResult> MatMulTensors(string tensorId, [FromBody] MatMulRequest request)
        {
            try
            {
                if (request.OtherData == null || request.OtherShape == null)
                {
                    return BadRequest("OtherData and OtherShape are required");
                }

                var grain = _clusterClient.GetGrain<ITensorGrain>(tensorId);
                var result = await grain.MatMul(request.OtherData, request.OtherShape);

                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error performing matrix multiplication for {TensorId}", tensorId);
                return StatusCode(500, "Internal server error");
            }
        }
    }

    public class TensorData
    {
        public float[]? Data { get; set; } = null;
        public int[]? Shape { get; set; } = null;
    }

    public class MatMulRequest
    {
        public float[]? OtherData { get; set; } = null;
        public int[]? OtherShape { get; set; } = null;
    }
}

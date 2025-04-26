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
    public class ComputeController : ControllerBase
    {
        private readonly ILogger<ComputeController> _logger;
        private readonly IClusterClient _clusterClient;

        public ComputeController(ILogger<ComputeController> logger, IClusterClient clusterClient)
        {
            _logger = logger;
            _clusterClient = clusterClient;
        }

        [HttpPost("{computeId}/forward")]
        public async Task<IActionResult> Forward(string computeId, [FromBody] ForwardRequest request)
        {
            try
            {
                if (request.ModelId == null || request.Input == null)
                {
                    return BadRequest("ModelId and Input are required");
                }

                var grain = _clusterClient.GetGrain<IComputeGrain>(computeId);
                var result = await grain.Forward(request.ModelId, request.Input);

                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error performing forward pass for {ComputeId}", computeId);
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpPost("{computeId}/backward")]
        public async Task<IActionResult> Backward(string computeId, [FromBody] BackwardRequest request)
        {
            try
            {
                if (request.ModelId == null || request.Gradients == null)
                {
                    return BadRequest("ModelId and Gradients are required");
                }

                var grain = _clusterClient.GetGrain<IComputeGrain>(computeId);
                var result = await grain.Backward(request.ModelId, request.Gradients);

                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error performing backward pass for {ComputeId}", computeId);
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpPost("{computeId}/update")]
        public async Task<IActionResult> UpdateModel(string computeId, [FromBody] UpdateModelRequest request)
        {
            try
            {
                if (request.ModelId == null || request.Parameters == null)
                {
                    return BadRequest("ModelId and Parameters are required");
                }

                var grain = _clusterClient.GetGrain<IComputeGrain>(computeId);
                var result = await grain.UpdateModel(request.ModelId, request.Parameters);

                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating model for {ComputeId}", computeId);
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpGet("{computeId}/parameters/{modelId}")]
        public async Task<IActionResult> GetModelParameters(string computeId, string modelId)
        {
            try
            {
                var grain = _clusterClient.GetGrain<IComputeGrain>(computeId);
                var parameters = await grain.GetModelParameters(modelId);

                return Ok(parameters);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting model parameters for {ComputeId}/{ModelId}", computeId, modelId);
                return StatusCode(500, "Internal server error");
            }
        }
    }

    public class ForwardRequest
    {
        public string? ModelId { get; set; } = null;
        public float[]? Input { get; set; } = null;
    }

    public class BackwardRequest
    {
        public string? ModelId { get; set; } = null;
        public float[]? Gradients { get; set; } = null;
    }

    public class UpdateModelRequest
    {
        public string? ModelId { get; set; } = null;
        public float[]? Parameters { get; set; } = null;
    }
}

class VllmConfig:
    def _get_v1_model_runner_unsupported_features(self) -> list[str]:
        unsupported: list[str] = []

        # PCP runtime support is implemented only by the V2 model runner.
        if self.parallel_config.prefill_context_parallel_size > 1:
            unsupported.append("prefill context parallel")

        # Note(arpera):
        # MRV1 + PP>1 + async sched + structured output
        # does not work in vLLM. For more info see:
        # https://github.com/vllm-project/vllm/issues/45014
        # Since recently MRV1 has been deprecated then
        # there was decided not to fix the issue but instead
        # to disallow such configuration.
        # At the same time MRV2 works fine in this case.
        if (
            self.parallel_config.pipeline_parallel_size > 1
            and self.scheduler_config.async_scheduling
        ):
            unsupported.append("pipeline parallelism with async scheduling")

        # DSpark is implemented only by the V2 GPU model runner.
        if self.speculative_config:
            # Imported here since it imports vllm.config.
            from vllm.v1.worker.gpu.spec_decode.xpress.config import (
                XPressSpeculatorConfig,
            )

            if self.speculative_config.method == "dspark":
                unsupported.append("dspark speculative decoding")
            elif isinstance(
                self.speculative_config.speculator_config, XPressSpeculatorConfig
            ):
                unsupported.append("xpress speculative decoding")
            if self.speculative_config.enable_adaptive_verification:
                unsupported.append("adaptive draft verification")
            if self.speculative_config.draft_lm_head_quantization is not None:
                unsupported.append("draft lm_head quantization")

        # The attention-free Parse MTP draft requires the V2 speculator.
        if self.speculative_config:
            draft_config = self.speculative_config.draft_model_config
            if draft_config is not None and "ParseAuxMTPModel" in (
                draft_config.architectures or []
            ):
                unsupported.append("Parse 3.5 MTP")

        # Mixed sliding/full DFlash drafts need multiple KV groups (V2 only).
        if self._dflash_needs_multi_kv_group():
            unsupported.append("mixed sliding/full dflash drafts")

        # DFlash candidate heads exist only in the V2 speculator. On
        # V1 the same checkpoint drafts through DFlashProposer, which never
        # calls it, so the draft would degrade to DFlash1 silently.
        if self._is_dflash_candidate_draft():
            unsupported.append("DFlash candidate-head drafts")

        if self.model_config is not None and self.model_config.is_diffusion:
            unsupported.append("diffusion models")

        if self.parallel_config.enable_batch_sharded_sampling:
            unsupported.append("batch-sharded sampling")

        if self.model_config is not None and self.model_config.architecture in (
            "Step5ForCausalLM",
            "Step5ForConditionalGeneration",
        ):
            unsupported.append("Step5")

        return unsupported

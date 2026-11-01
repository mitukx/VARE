# Task: Preserve CSM codebook embedding tying configuration

Repair the CSM configuration behavior in `configuration_csm.py` so the model's codebook embedding-tie setting is controlled consistently by `tie_codebooks_embeddings`. The behavior must work both when codebook tying is enabled and disabled, including when loading a serialized `tie_word_embeddings` field. Keep text embeddings and the language-model head untied. Preserve the configuration API and other defaults.

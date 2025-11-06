1. Activate current lerobot env, then in so101-exp/, run \
    ``` pip install -e .
    pip install packaging ninja 
    ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
    pip install "flash-attn==2.5.5" --no-build-isolation ```
3. put ``policies/adapter`` to ``lerobot/src/lerobot/policies/adapter``
4. replace ``lerobot/src/lerobot/policies/__init__.py, factory.py`` with ``policies/__init__.py, factory.py``
5. Download https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b to ``lerobot/src/lerobot/policies/adapter/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b``, and https://huggingface.co/daisy-zzz/pickcube_so101/tree/main/step_20000 to ``lerobot/src/lerobot/policies/adapter/checkpoints``, then use command ``hf auth login`` in for further huggingface downloading during inference
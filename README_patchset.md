# Patchset Pi 5 / Trixie / V3DV (inspiré des observations Dolphin)

## Fichiers à utiliser
- `vk_platform.cpp`
- `vk_instance.cpp`
- `vk_graphics_pipeline.cpp`
- `vk_rasterizer.cpp`

## Pourquoi ces fichiers
1. `vk_platform.cpp`
   - coupe `VK_LAYER_KHRONOS_validation` au bon endroit, à la création d'instance.

2. `vk_instance.cpp`
   - restaure le flux normal du constructeur.
   - la désactivation de validation ne doit pas être forcée ici.

3. `vk_graphics_pipeline.cpp`
   - force `sampleShadingEnable = false`.
   - c'est le correctif le plus important restant d'après les logs historiques.

4. `vk_rasterizer.cpp`
   - garde le fichier compilable.
   - supprime les appels invalides `Submit()/WaitIdle()` qui cassaient l'installation.

## Lecture du dernier log
- la validation Vulkan est bien supprimée à la création d'instance.
- le backend Vulkan initialise correctement V3DV.
- le prochain risque connu reste le pipeline multisample / sample shading.

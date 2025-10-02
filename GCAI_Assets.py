import os
from GCAI_Exceptions import MissingAssetError
from GCAI_Types import AssetType

class GCAI_AssetCollector:
    """
        Collects and validates paths to required assets for the application.
        
        Throws MissingAssetError if paths to assets are invalid.
    """
    def __init__(self) -> None:
        self._asset_dir: str = self._get_asset_dir()
        self._model_path: str = self._find_asset_path("model.hef", AssetType.MODEL)
        self._models: dict[str, str] = {
            "single": self._find_asset_path("model_single.hef", AssetType.MODEL),
            "multi": self._find_asset_path("model_multi.hef", AssetType.MODEL)
        }
        self._postprocess_path: str = self._find_asset_path("postprocess.so", AssetType.POSTPROCESS)
        # Other assets can be added here
        
    def _get_asset_dir(self) -> str:
        root_dir = os.path.dirname(__file__)
        asset_dir = os.path.join(os.path.dirname(root_dir), "res")

        if not os.path.isdir(asset_dir):
            raise MissingAssetError(AssetType.DIRECTORY, asset_dir)
        return asset_dir
    
    def _find_asset_path(self, asset_name: str, asset_type: AssetType) -> str:
        asset_path = os.path.join(self._asset_dir, asset_name)

        if not os.path.isfile(asset_path):
            raise MissingAssetError(asset_type, asset_path)
        return asset_path
    
    def get_model(self, model_type: str) -> str:
        if model_type not in self._models:
            raise MissingAssetError(AssetType.MODEL)
        return self._models[model_type]
    
    @property
    def postprocess_so(self) -> str:
        return self._postprocess_path
    
if __name__ == "__main__":
    collector = GCAI_AssetCollector()

    
    
from GCAI_Types import AssetType

class MissingAssetError(Exception):
    """
        Exception raised when critical assets needed for the 
        program to work are missing.
    """

    def __init__(self, asset_type: AssetType, asset_path: str = "unknown"):
        message: str = ""
        if asset_type == AssetType.MODEL and asset_path == "unknown":
            message = "Incorrect model type provided. Should be 'single' or 'multi'."

        message = f"{asset_type} not found. Path {asset_path} could be invalid."
        super().__init__(message)
        self.type = asset_type
        self.path = asset_path
class CreativeMock:
    def explain(self):
        return "mock explanation"

def get_best_creatives(creative_id, format_type, theme, hook):
    """
    Retrieve top cases.
    """
    return [CreativeMock(), CreativeMock()]

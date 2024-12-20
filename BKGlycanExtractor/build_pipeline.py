from BKGlycanExtractor import Config_Manager, GlycanExtractorPipeline

class BuildPipeline:

    def __init__(self,pipeline_name,end_step):
        self.pipeline_name = pipeline_name
        self.end_step = end_step  
        
        self.config = Config_Manager()    


    def load_pipeline(self):

        pipeline_steps = self.config.get_pipeline(self.pipeline_name)
        glycan_steps = pipeline_steps.get_steps('glycan')

        # glycan_steps = self.resolve_pipeline_steps(glycan_steps, self.end_step)
        known_steps, end_known_step = self.resolve_pipeline_steps(glycan_steps)

        pipeline = GlycanExtractorPipeline()

        pipeline.steps['figure'] = pipeline_steps.get_steps('figure')
        pipeline.steps['glycan'] = known_steps

        return pipeline, end_known_step


    def resolve_pipeline_steps(self, steps):
        classes = [step.__class__.__name__ for step in steps] 

        end_pred_step = self.config.get_finder(self.end_step)
        end_step_class = end_pred_step.__class__.__name__ 


        end_known_step = end_pred_step.known_predictor()
        known_steps = [step.known_predictor() for step in steps]

        # ensuring all the steps (except the last step) are present for the base pipeline - which
        # will be used for known_pipeline and pred_pipeline both - and later the known_end_step and
        # pred_end_step will be applied separately
        if end_step_class in classes:
            idx = classes.index(end_step_class)
            steps = steps[:idx]
            # end_known_step = known_steps[idx]
            known_steps = known_steps[:idx] 
        
        return known_steps, end_known_step
        
from .imports import *
from .model_holder import ModelHolder
from .data_holder import DataHolder
from .misc import softmax, find_top_indexes
from .processor_helper import ProcessorHelper

class Sync():
    def __init__(self, config=None):
        self.mhold = None
        self.dhold = None
        self.phelp = ProcessorHelper()
        self.config = config

    def prep_gen_inputs(self):
        args = self.dhold.inputs
        self.dhold.gen_inputs = args

        self.phelp.load_model(self)
        self.phelp.load_beam_config(self)
        self.phelp.build_prompt_string(self)
        self.phelp.prepare_model_generation_args(self)

        

    def get_best_path(self):

        args = self.dhold.gen_inputs

        self.dhold.total_probs  = []
        self.dhold.prediction_paths_probs = []
        self.dhold.prediction_paths_indices = []
        self.dhold.skip_path = []

        self.dhold.logits_merker = copy.deepcopy(self.dhold.logits)
        self.dhold.considered_tokens_num_merker = copy.deepcopy(self.dhold.considered_tokens_num)

        self.phelp.beamsearch_setup_inputs(self)

        self.phelp.beamsearch_do_inference(self)

        self.phelp.beamsearch_get_beams_from_outputs(self)
        self.phelp.beamsearch_get_best_beam_from_beams(self)



    def do_inference(self, limit_tokens=None, alternative_input=None, alternative_mask=None, llama_sequencial_batch=False):
        self.dhold.start_time_inference = time.time()
        self.dhold.limit_tokens = limit_tokens
        self.dhold.alternative_input = alternative_input
        self.dhold.alternative_mask = alternative_mask
        self.dhold.llama_sequencial_batch = llama_sequencial_batch

        self.phelp.check_for_error_and_limit_tokens(self)

        
        
        self.phelp.inference_setup_args(self)
        self.phelp.inference_check_stop_token_and_alternative_inputs(self)

        self.phelp.inference_do_inference(self)
        self.phelp.inference_get_considered_tokens_num(self)

        if self.dhold.inputs['debugmode']: print("self.dhold.returned_content:", self.dhold.returned_content)



    # sets dhold.returned_content, dhold.output_shape, self.dhold.logits (and maybe dhold.input_shape)
    def generate(self):
        
        # if normal
        # generate no limit
        if not self.dhold.inputs['beam_config']['use_beam_search']:
            self.do_inference()
        else:
            self.dhold.generated_tokens = 0
            while self.dhold.generated_tokens < self.dhold.inputs['max_new_tokens']:
                # generate limit 1 token
                self.do_inference(limit_tokens=1)

                self.phelp.beamsearch_do_search(self)
                self.dhold.generated_tokens += len(self.dhold.tokens_to_add)
                self.phelp.append_tokens_to_add_to_tokens(self)
                
                self.phelp.print_beam_debug_info(self)
                
                self.phelp.beamsearch_check_break_condition(self)
                if self.dhold.beamsearch_break:
                    break

            # end
            self.phelp.beamsearch_get_returned_content(self)
            


    
    def make_new_dhold(self):
        self.dhold = DataHolder()

    def make_new_mhold(self):
        self.mhold = ModelHolder()
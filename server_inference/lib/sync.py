from .imports import *
from .model_holder import ModelHolder
from .model_holder_image import ModelHolder as ModelHolderImageGen
from .data_holder import DataHolder
from .misc import softmax, find_top_indexes
from .processor_helper import ProcessorHelper

class Sync():
    def __init__(self, config=None, mode=None):
        self.mhold = None
        self.dhold = None
        self.mode = mode
        self.phelp = ProcessorHelper()
        self.config = config

    def generate_image(self):
        if self.mhold == None or self.mhold.current_model != self.dhold.inputs['model']:
            self.mhold = ModelHolderImageGen()
            self.mhold.load_model(self, self.dhold.inputs['model'], "float16")

        image = self.mhold.model(self.dhold.inputs['prompt']).images[0] 
        self.dhold.generated_image = [image]

        

    def prep_gen_inputs(self):
        """
        prepares sync for a generation run, has to be used 
        at the beginning no matter which mode.

        loads model
        loads beam config
        builds prompt string
        sets generation args based on model
        """
        
        args = self.dhold.inputs
        self.dhold.gen_inputs = args

        self.phelp.load_model(self)
        self.phelp.load_beam_config(self)
        self.phelp.build_prompt_string(self)
        self.phelp.prepare_model_generation_args(self)

        

    def get_best_path(self):
        """
        gets the best path by conducting a beam search:
        sets up beam search inputs
        does inference
        gets beams from generation outputs
        gets best beam from beams
        """

        args = self.dhold.gen_inputs
        self.phelp.beamsearch_setup_inputs(self)
        self.phelp.beamsearch_do_inference(self)
        self.phelp.beamsearch_get_beams_from_outputs(self)
        self.phelp.beamsearch_get_best_beam_from_beams(self)



    def do_inference(self, limit_tokens=None, alternative_input=None, alternative_mask=None, sequencial_batch=False):
        """
        does inference with the information saved in sync.dhold:
        checks for error and token limit
        sets up generation arguments
        makes sure stop token is set and checks for alternative inputs
        runs generation
        if beam search: gets number of considered tokens
        """
        
        self.dhold.start_time_inference = time.time()
        self.dhold.limit_tokens = limit_tokens
        self.dhold.alternative_input = alternative_input
        self.dhold.alternative_mask = alternative_mask
        self.dhold.sequencial_batch = sequencial_batch

        self.phelp.inference_check_for_error_and_limit_tokens(self)

        
        
        self.phelp.inference_setup_args(self)
        self.phelp.inference_check_stop_token_and_alternative_inputs(self)

        self.phelp.inference_do_inference(self)
        if self.dhold.error:
            self.dhold.beamsearch_break = True # just in case of beam-search
            return None

        if self.dhold.inputs['beam_config']['use_beam_search']:
            self.phelp.inference_get_considered_tokens_num(self)

        if self.dhold.inputs['debugmode']: print("self.dhold.returned_content:", self.dhold.returned_content)



    # sets dhold.returned_content, dhold.output_shape, self.dhold.logits (and maybe dhold.input_shape)
    def generate(self):
        """
        main function for all types of generation
        """
        
        gc.collect()
        torch.cuda.empty_cache()
        self.prep_gen_inputs()

        with torch.autograd.grad_mode.inference_mode():
            
            # if normal
            # generate no limit
            if not self.dhold.inputs['beam_config']['use_beam_search']:
                self.do_inference()
            else:
                self.dhold.generated_tokens = 0
                while self.dhold.generated_tokens < self.dhold.inputs['max_new_tokens']:
                    # generate limit 1 token
                    self.do_inference(limit_tokens=1)
                    if hasattr(self.dhold, "beamsearch_break") and self.dhold.beamsearch_break:
                        break
    
                    self.phelp.beamsearch_do_search(self)
                    self.dhold.generated_tokens += len(self.dhold.tokens_to_add)
                    self.phelp.append_tokens_to_add_to_tokens(self)
                    
                    self.phelp.print_beam_debug_info(self)
                    
                    self.phelp.beamsearch_check_break_condition(self)
                    if self.dhold.beamsearch_break:
                        break
    
                # end
                self.phelp.beamsearch_get_returned_content(self)



    def solve_agent_task(self):
        """
        splits task into several steps to process with agents
        """
        
        base_instruction = "You are part of a network of agents where each agent has its own task, in order to handle complex queries. Your task is to "
        self.phelp.build_task(
            self, 
            instruction = base_instruction+"break down the users query into small chunks of facts. Do not solve the query. Only write the facts that are explicitly stated in the users query. After extracting the facts, state the task that the users query contains. Example:\nfact 1: ...\nfact 2: ...\n...\nquery: ...",
            information_input=None
        )
        self.generate()
        result_0 = self.dhold.returned_content[0]
        print(f"\n\n\nresult:\n\"{result_0}\"")

        self.phelp.build_task(
            self, 
            instruction = base_instruction+"Infer new facts based on the provided facts, if possible. If no correct new facts can be infered, only write the given facts. Do not solve the query. Think step by step while solving your task and always lay out your thoughts to avoid making mistakes or incorrect statements. Before writing down something as a statement, visualize it symbolically to make sure your logic is sound. Example:\nfact 1: ...\nfact 2: ...\n...\nquery: ...",
            information_input=result_0
        )
        self.generate()
        result_1 = self.dhold.returned_content[0]
        print(f"\n\n\nresult:\n\"{result_1}\"")

        self.phelp.build_task(
            self, 
            instruction = base_instruction+"make sure another agents task was completed correctly and no mistakes were made. In the following you will see the information given to this agent, its instruction and its output. Do not follow the other agents instruction, only follow this instruction: Before deciding on a rating, Think step by step through the other agents logic so make sure there are no flaws that are not noticable on a quick look. Always lay out your thoughts at every step. After you went through everything, rate the agents output from 0 to 9 on correctness, where 0 means that mistakes were made or something incorrect was written, and 9 means that the task was solved perfectly. ",
            information_input="given information to the agent: "+result_0+"\ninstruction: \""+base_instruction+'Infer new facts based on the provided facts, if possible. If no correct new facts can be infered, only write the given facts. Do not solve the query. Example:\nfact 1: ...\nfact 2: ...\n...\nquery: ...'+"\"\nagents result: "+result_1
        )
        self.generate()
        result_2 = self.dhold.returned_content[0]
        print(f"\n\n\nresult:\n\"{result_2}\"")
            


    def make_new_dhold(self):
        self.dhold = DataHolder()

    def make_new_mhold(self):
        self.mhold = ModelHolder()

    def change_mode(self, new_mode):
        self.__init__(mode=new_mode, config=self.config)
        gc.collect()
        torch.cuda.empty_cache()
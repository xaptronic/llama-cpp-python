from llama_cpp import Llama

grammar = """
root      ::= nav eol (commands eol)*
commands  ::= t | info
nav       ::= "nav(\\"admin/" [a-z/]*  "\\")"
info      ::= "info(" setting ")"
t         ::= "t(" setting ", " value ")"
value     ::= color | string | number | boolean
color     ::= "#" [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]
setting   ::= "\\"" [a-z ]+ "\\""
string    ::= "\\"" [ \\t!#-\\[\\]-~]* "\\""
number    ::= [0-9]+
boolean   ::= ("true" | "false")
eol       ::= "\\n"
"""

llm = Llama(
    model_path="/Users/alex/llama-7b.ggmlv3.q8_0.bin",
    lora_base="/Users/alex/llama-7b.ggml.f16.bin",
    # python ~/llama.cpp/convert-lora-to-ggml.py .
    lora_path="/Users/alex/src/github.com/Shopify/sidekick-data/src/webapp/models/ggml-adapter-model.bin",
    # n_gpu_layers=1000,
    n_ctx=2048,
    grammar=grammar,
)

# response = llm("make my theme orange")

import code
code.interact(local=globals())

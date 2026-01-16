Thought process so far for solving the execution agent overload issue:
- the strategies mentioned in article; semantic search over descriptions, archiving dormant agents, clustering agents, maintining hot cache of recently active agents, all seem like good solutions to the issue and are what I would have thought of first
- initial idea was to implement a LRU cache of active agents and also use semantic search on agent descriptions as mentioned to get relevant agents as well
- I implemented LRU cache first and tested as it was the simpler of the two to implement


- I swapped to using gpt-oss-120b as it was a free model on OpenRouter, and this reinforced that agent descriptions were needed for semantic search
    - idea was to generate agent descriptions by passing in a filtered transcript of the agent with a prompt to summarize its purpose
- I then realized that the agent names being created were very generic, things like "email search" and "find email", confused me because agent names in article were more descriptive
- loaded some credits on OpenRouter and switched to the default claude sonnet 4 model the app was built with and names were descriptive
- I realized semantic search would likely be better on agent names in this case
- This led me to think about how in deployment, different users would use different models for this agent based on cost, speed, etc, so implementing descriptions would still be useful and I could make it a configurable option


- for sake of time I focused on implementing semantic search over agent names first
- during this process, I realized that semantic search using user query over agent names could cause problems when the user query had multiple names in it, example I used to realize this was:
    - previous message was something like "did alice reply"
    - latest user message was "what about ryan or vince"
- in this case, semantic search was assigning high relevance to the "Email To Ryan" agent, but not to the "Vince Email Search" agent that existed
- semantically, there was much more weight on the term "ryan" as it came first in the query
- my initial thought to solve this was to switch to a hybrid search that boosted semantic scores for keyword matches
- although this would likely work for this example, it got me thinking further about how the interaction agent maintains context
  - another example is if I had a request earlier that was "send an email to alice" and then later I say "did she reply", semantic search AND keyword search wouldn't be useful here
- my first thought to fix this was to make the semantic search not just use the user query but a few previous turns in the conversation
  - this would be difficult then to keep more relevance on the latest message in the conversation, I needed to make the previous messages just be context
- the simplest way I thought to solve this context problem as well as the semantic problem when a user query had multiple names or multiple intents, was to use llm decomposition
- I created a function that would take the user request and the last few messages in the conversation and made a prompt that instructed the llm to create queries for the purpose of semantic search
  - the focus for creating the queries is to capture relevant entities and intents from the user request using previous history as context to resolve pronouns, etc
  - this seems to work great so far in testing
- I currently have hot list size set to 10 agents, and top k for semantic search set to 15 agents
  - I think this is a good size that still reduces prompt bloat, while providing enough relevant agents for the interaction agent to choose from, assuming the number of agents would be 100s to 1000s for a user
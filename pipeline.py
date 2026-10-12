from agents import build_reader_agent,build_search_agent,writer_chain,critic_chain
def run_research_pipeline(topic: str )-> dict:

    state ={}

    #search agent working
    print("\n"+"= "*50)
    print("step 1 - search agent is working...")
    print("= "*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages":[("user",f"Find recent and reliable information on the topic: {topic}.")]

    })
    state['search_results'] = search_result['messages'][-1].content

    print("\n search result",state['search_results'])

    #step 2 - reader agent working
    print("\n"+"= "*50)
    print("step 2 - reader agent is working...")
    print("= "*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages":[("user",f"Based on the following search results about '{topic}',"
                      f"pick the most relevant URLs and scrape their content.\n\nSearch Results:\n{state['search_results']}\n\n"
                      f"Search Results:\n{state['search_results'][:800]}")]

    })
    state['scraped_content'] = reader_result['messages'][-1].content

    print("\n reader result",state['scraped_content'])
    #step 3 - writer chain working
    print("\n"+"= "*50)
    print("step 3 - writer chain is working...")
    print("= "*50)

    research_combined = (
        f"Search Results:\n{state['search_results']}\n\n"
        f"detailed scraped content:\n{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined  
    })

    print("\n Final Report:\n", state["report"])

    #critic report
    print("\n"+"= "*50)
    print("step 4 - critic chain is working...")
    print("= "*50)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\n Critic Feedback:\n", state["feedback"])

    return state


if __name__ == "__main__":
    topic=input("\n Enter the research topic: ")
    run_research_pipeline(topic)


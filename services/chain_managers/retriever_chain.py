from langchain.chains.retrieval_qa.base import RetrievalQA


class RetrieverChainManager:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.retriever_chain = None

    def get_retriever_chain(self):
        if self.retriever_chain is None:
            self.retriever_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                return_source_documents=True,
                chain_type="refine"
            )
        return self.retriever_chain
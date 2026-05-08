import re
from collections.abc import AsyncGenerator, Awaitable
from dataclasses import asdict
from typing import Any, Optional, cast

from azure.search.documents.aio import SearchClient
from azure.search.documents.knowledgebases.aio import KnowledgeBaseRetrievalClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI, AsyncStream
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseCompletedEvent,
    ResponseOutputMessage,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)

from approaches.approach import (
    Approach,
    ExtraInfo,
    ThoughtStep,
)
from approaches.promptmanager import PromptManager
from prepdocslib.blobmanager import AdlsBlobManager, BlobManager
from prepdocslib.embeddings import ImageEmbeddings

from lxml import etree

from pm4py.objects.dcr.utils.utils import nested_groups_and_sps_to_flat_dcr as flatten_dcr
from pm4py.objects.dcr.semantics import DcrSemantics
from pm4py.objects.dcr.importer import importer as dcr_importer

class LawLLM():
    def __init__(self):
        pass

    def call(message:str):
        answer = ""
        return answer
    
class DocumentingLLM():
    def __init__(self):
        pass

    def call(message:str):
        answer = ""
        return answer

class SimilarCasesLLM():
    def __init__(self):
        pass

    def call(message:str):
        answer = ""
        return answer

class DcrLLM():
    '''Orchestrating LLM'''
    def __init__(self):
        pass

    def call(message:str):
        answer = ""
        return answer

def remove_last_line(s: str) -> str:
    lines = s.splitlines()
    return "\n".join(lines[:-1]) if len(lines) > 1 else ""

def starts_xml(text: str) -> bool:
    return text.startswith('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')

def is_xml_parsing(text: str) -> bool:
    try:
        etree.fromstring(text.encode())
        return True
    except etree.XMLSyntaxError:
        return False

simple_decision_events = {'Is child under 18': lambda x : x < 18}

llm_decision_events = {
    'Resonable expenses': SimilarCasesLLM,
    'merudgifterne er en konsekvens af den nedsatte funktionsevne': SimilarCasesLLM,
    'lidelse og ikke kan dækkes efter andre bestemmelser i denne lov eller anden lovgivning': LawLLM,
    'nedsat fysisk eller psykisk funktionsevne': SimilarCasesLLM,
}

llm_case_summary_events = {
    'Cover expenses': DocumentingLLM,
    'Do not cover': DocumentingLLM,
}

# data_request_events = {
#     'Add expenses': 'Please add the expense Amount and Description.', 
#     'Consequence of disability': 'Please document the consequence of disability', 
#     'Age': 'Please add your age',  
#     'Not covered by other laws': 'Have you requested expenses that are covered by other laws?', 
#     'Significant or permanent disability': 'Please document whether you have a significant or permanent disability.'
# }

from quart import current_app

class ChatReadRetrieveReadApproach(Approach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then uses Azure AI Search to retrieve relevant documents, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """


    NO_RESPONSE = Approach.QUERY_REWRITE_NO_RESPONSE

    def __init__(
        self,
        *,
        search_client: SearchClient,
        search_index_name: str,
        knowledgebase_model: Optional[str],
        knowledgebase_deployment: Optional[str],
        knowledgebase_client: Optional[KnowledgeBaseRetrievalClient],
        knowledgebase_client_with_web: Optional[KnowledgeBaseRetrievalClient] = None,
        knowledgebase_client_with_sharepoint: Optional[KnowledgeBaseRetrievalClient] = None,
        knowledgebase_client_with_web_and_sharepoint: Optional[KnowledgeBaseRetrievalClient] = None,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        embedding_dimensions: int,
        embedding_field: str,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
        prompt_manager: PromptManager,
        reasoning_effort: Optional[str] = None,
        multimodal_enabled: bool = False,
        image_embeddings_client: Optional[ImageEmbeddings] = None,
        global_blob_manager: Optional[BlobManager] = None,
        user_blob_manager: Optional[AdlsBlobManager] = None,
        use_web_source: bool = False,
        use_sharepoint_source: bool = False,
        retrieval_reasoning_effort: Optional[str] = None,
    ):
        self.search_client = search_client
        self.search_index_name = search_index_name
        self.knowledgebase_model = knowledgebase_model
        self.knowledgebase_deployment = knowledgebase_deployment
        self.knowledgebase_client = knowledgebase_client
        self.knowledgebase_client_with_web = knowledgebase_client_with_web
        self.knowledgebase_client_with_sharepoint = knowledgebase_client_with_sharepoint
        self.knowledgebase_client_with_web_and_sharepoint = knowledgebase_client_with_web_and_sharepoint
        self.openai_client = openai_client
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.embedding_field = embedding_field
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.prompt_manager = prompt_manager
        self.query_rewrite_tools = self.prompt_manager.load_tools("chat_query_rewrite_tools.json")
        self.reasoning_effort = reasoning_effort
        self.include_token_usage = True
        self.multimodal_enabled = multimodal_enabled
        self.image_embeddings_client = image_embeddings_client
        self.global_blob_manager = global_blob_manager
        self.user_blob_manager = user_blob_manager
        # Track whether web source retrieval is enabled for this deployment; overrides may only disable it.
        self.web_source_enabled = use_web_source
        self.use_sharepoint_source = use_sharepoint_source
        self.retrieval_reasoning_effort = retrieval_reasoning_effort

        self.dcr_graph = None
        self.semantics = None
        self.enabled_events = set()
        self.citizen_data = {}
        self.event_role_mapping = {}

    def extract_followup_questions(self, content: Optional[str]):
        if content is None:
            return content, []
        return content.split("<<")[0], re.findall(r"<<([^>>]+)>>", content)

    def get_search_query(self, response: Response, default_query: str) -> str:
        """Read the optimized search query from a response tool call."""
        try:
            return self.extract_rewritten_query(response, default_query, no_response_token=self.NO_RESPONSE)
        except Exception:
            return default_query

    async def execute_dcr_event(self, event):
        self.dcr_graph = self.semantics.execute(self.dcr_graph, event)
        is_graph_accepting = self.semantics.is_accepting(self.dcr_graph)
        self.enabled_events = self.semantics.enabled(self.dcr_graph)
        content = f"enabled: {self.enabled_events} \n\n pending: {self.dcr_graph.marking.pending} \n\n accepting: {is_graph_accepting}"
        return content

    async def get_event_requirements(self, event):
        role_for_event = self.event_role_mapping[event]

        if role_for_event == 'Citizen Data':
            #here I have to collect data from the citizen by asking questions 
            # and recording the answer in citizen data
            pass
        elif role_for_event in ['Robot','Case Management System', 'Decision']:
            if event in simple_decision_events.keys():
                #these execute by inspecting the existing data or
                #if the data is not found then it is requested from the user
                pass
            elif event in llm_decision_events.keys():
                #these require a search in the documents llm. or
                #these require a retrieval of historical cases llm
                clarifying_prompt  = f"""Given this event {event}, this data {self.citizen_data} and this dcr graph {self.dcr_graph}. 
                Does the data allow you to execute the event? 
                Answer yes/no only!"""
                answer_interpreted = self.run_just_llm(clarifying_prompt)
                pass
            elif event in llm_case_summary_events.keys():
                #these require a case summary llm
                pass

    async def parse_dcr_from_answer(self,content):
        answer = content
        while not is_xml_parsing(answer):
            answer = remove_last_line(answer)

        self.dcr_graph = flatten_dcr(dcr_importer.deserialize(answer))
        self.semantics = DcrSemantics()
        self.enabled_events = self.semantics.enabled(self.dcr_graph)
        is_graph_accepting = self.semantics.is_accepting(self.dcr_graph)
        content = f"enabled: {self.enabled_events} \n\n pending: {self.dcr_graph.marking.pending} \n\n accepting: {is_graph_accepting}"
        
        original_dict = self.dcr_graph.role_assignments
        self.event_role_mapping = {value: key for key, value_set in original_dict.items() for value in value_set}

        return content

    async def run_without_streaming(
        self,
        messages: list[EasyInputMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> dict[str, Any]:
<<<<<<< HEAD
                
        original_user_query = messages[-1]["content"]
        # previous_agent_question = ""
        # if len(messages)>1:
        #     previous_agent_question = messages[-2]["content"]

        # if previous_agent_question == "What is the child's age?":
        #     # current_app.logger.info(original_user_query)
        #     # initial_prompt = f"""
        #     # If there is a number here: {original_user_query} extract that number and only return the number! 
        #     # Otherwise return -1"""
        #     # chat_app_response = await self.run_just_llm(initial_prompt,messages)
        #     age = int(original_user_query)
        #     self.citizen_data['Age'] = age

        if self.dcr_graph and isinstance(original_user_query, str) and original_user_query in self.dcr_graph.events:
            # This handles the DCR related answers
            event = original_user_query

        #     role_for_event = self.event_role_mapping[event]
        #     if event == 'Age':
        #         chat_app_response = {
        #             "message": {"content": "What is the child's age?", "role": "agent"},
        #             "context": {
        #                 "thoughts": [],
        #                 "data_points": {},
        #                 "followup_questions": list(self.enabled_events),
        #             },
        #             "session_state": session_state,
        #         }
        #         return chat_app_response
        #     elif event == 'Is child under 18':
        #         result = simple_decision_events[event](self.citizen_data['Age'])
        #         current_app.logger.info(result)
        #         initial_prompt = f"""
        #         User was asked: {event}? Answer was: {result}
        #         Can the user proceed?
        #         """
        #         chat_app_response = await self.run_just_llm(initial_prompt)
        #         return chat_app_response
        #     elif role_for_event == 'Citizen Data' and event not in self.citizen_data:
        #         initial_prompt = event
        #         initial_prompt = f"""
        #         Ask the user to provide the data or documentation about this event: '{initial_prompt}'.
        #         Make it into a precise question that can be answered by the user!
        #         For example if the event is Age say: What is the childs age?
        #         """
        #         chat_app_response = await self.run_just_llm(initial_prompt)
        #         current_app.logger.info(chat_app_response)
        #         chat_app_response["context"]["followup_questions"] = list(self.enabled_events)
        #         return chat_app_response
        #     else:
        #         self.get_event_requirements(event)
            content = await self.execute_dcr_event(event)
            chat_app_response = {
                "message": {"content": content, "role": "agent"},
                "context": {
                    "thoughts": [],
                    "data_points": {},
                    "followup_questions": list(self.enabled_events),
=======
        extra_info, response_coroutine = await self.run_until_final_call(
            messages, overrides, auth_claims, should_stream=False
        )
        response: Response = await cast(Awaitable[Response], response_coroutine)
        content = response.output_text
        if overrides.get("suggest_followup_questions"):
            content, followup_questions = self.extract_followup_questions(content)
            extra_info.followup_questions = followup_questions
        if self.include_token_usage and extra_info.thoughts and response.usage:
            extra_info.thoughts[-1].update_token_usage(response.usage)
        chat_app_response = {
            "output_text": content,
            "context": {
                "thoughts": extra_info.thoughts,
                "data_points": {
                    key: value for key, value in asdict(extra_info.data_points).items() if value is not None
>>>>>>> 95ce0c9484b338b3819914d0c1a1fa8d19a3ff9b
                },
                "session_state": session_state,
            }
            return chat_app_response

        else:
            # this handles the non dcr events answers!
            extra_info, chat_coroutine = await self.run_until_final_call(
                messages, overrides, auth_claims, should_stream=False
            )
            chat_completion_response: ChatCompletion = await cast(Awaitable[ChatCompletion], chat_coroutine)
            content = chat_completion_response.choices[0].message.content
            role = chat_completion_response.choices[0].message.role

            if starts_xml(content):
                # If the LLM returns a dcr graph then parse it and start dcr execution
                content = await self.parse_dcr_from_answer(content)

                if overrides.get("suggest_followup_questions"):
                    extra_info.followup_questions = list(self.enabled_events)
            else:
                # these are the normal ansers
                dcr_trigger = "Return the dcr graph .xml file for this paragraph!"
                if content == dcr_trigger:
                    dcr_trigger += " Return only the .xml content!!!!"

                if overrides.get("suggest_followup_questions"):
                    content, followup_questions = self.extract_followup_questions(content)
                    followup_questions.append(dcr_trigger)
                    followup_questions.append(list(self.enabled_events))
                    extra_info.followup_questions = followup_questions

            # Assume last thought is for generating answer
            # TODO: Update for agentic? This isn't still true?
            if self.include_token_usage and extra_info.thoughts and chat_completion_response.usage:
                extra_info.thoughts[-1].update_token_usage(chat_completion_response.usage)
            chat_app_response = {
                "message": {"content": content, "role": role},
                "context": {
                    "thoughts": extra_info.thoughts,
                    "data_points": {
                        key: value for key, value in asdict(extra_info.data_points).items() if value is not None
                    },
                    "followup_questions": extra_info.followup_questions,
                },
                "session_state": session_state,
            }
            return chat_app_response

    async def run_just_llm(
        self,
        message,
        past_messages: Any = None, 
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        q = message
        if not isinstance(q, str):
            raise ValueError("The most recent message content must be a string.")

        else:
            current_app.logger.info(past_messages)
            # Process results
            messages = self.prompt_manager.render_prompt(
                self.prompt_manager.load_prompt("basic.prompty"),
                {
                    "user_query": q,
                    "past_messages": past_messages,
                    "text_sources": "",
                    "image_sources": [],
                    "citations": [],
                },
            )

            chat_completion = cast(
                ChatCompletion,
                await self.create_chat_completion(
                    self.chatgpt_deployment,
                    self.chatgpt_model,
                    messages=messages,
                    overrides=overrides,
                    response_token_limit=self.get_response_token_limit(self.chatgpt_model, 1024),
                ),
            )

            answer = chat_completion.choices[0].message.content or ""

        return {
            "message": {
                "content": answer,
                "role": "assistant",
            },
            "context": {
                "thoughts": [],
                "data_points": {},
            },
            "session_state": session_state,
        }

    async def run_with_streaming(
        self,
        messages: list[EasyInputMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> AsyncGenerator[dict, None]:
<<<<<<< HEAD
        
        extra_info, chat_coroutine = await self.run_until_final_call(
=======
        extra_info, response_coroutine = await self.run_until_final_call(
>>>>>>> 95ce0c9484b338b3819914d0c1a1fa8d19a3ff9b
            messages, overrides, auth_claims, should_stream=True
        )
        yield {"type": "response.context", "context": extra_info, "session_state": session_state}

        followup_questions_started = False
        followup_content = ""
        result = await response_coroutine

        # Handle non-streaming Response (e.g. when agentic retrieval already provided an answer)
        if isinstance(result, Response):
            content = result.output_text or ""

            followup_questions: list[str] = []
            if overrides.get("suggest_followup_questions"):
                content, followup_questions = self.extract_followup_questions(content)
                extra_info.followup_questions = followup_questions

            if self.include_token_usage and extra_info.thoughts and result.usage:
                extra_info.thoughts[-1].update_token_usage(result.usage)

            if content:
                yield {"type": "response.output_text.delta", "delta": content}

            yield {"type": "response.context", "context": extra_info, "session_state": session_state}
            return

        # Handle streaming Response events
        stream = cast(AsyncStream[ResponseStreamEvent], result)

        async for event in stream:
            if isinstance(event, ResponseTextDeltaEvent):
                delta_content: str = event.delta or ""
                if overrides.get("suggest_followup_questions") and "<<" in delta_content:
                    followup_questions_started = True
                    earlier_content = delta_content[: delta_content.index("<<")]
                    if earlier_content:
                        yield {"type": "response.output_text.delta", "delta": earlier_content}
                    followup_content += delta_content[delta_content.index("<<") :]
                elif followup_questions_started:
                    followup_content += delta_content
                else:
                    yield {"type": "response.output_text.delta", "delta": delta_content}
            elif isinstance(event, ResponseCompletedEvent):
                if event.response.usage and extra_info.thoughts and self.include_token_usage:
                    extra_info.thoughts[-1].update_token_usage(event.response.usage)
                    yield {"type": "response.context", "context": extra_info, "session_state": session_state}

        if followup_content:
            _, followup_questions = self.extract_followup_questions(followup_content)
            extra_info.followup_questions = followup_questions
            yield {"type": "response.context", "context": extra_info, "session_state": session_state}

    async def run(
        self,
        messages: list[EasyInputMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        return await self.run_without_streaming(messages, overrides, auth_claims, session_state)

    async def run_stream(
        self,
        messages: list[EasyInputMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> AsyncGenerator[dict[str, Any], None]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        return self.run_with_streaming(messages, overrides, auth_claims, session_state)

    async def run_until_final_call(
        self,
        messages: list[EasyInputMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[ExtraInfo, Awaitable[Response] | Awaitable[AsyncStream[ResponseStreamEvent]]]:
        use_agentic_knowledgebase = True if overrides.get("use_agentic_knowledgebase") else False
        original_user_query = messages[-1]["content"]

        if use_agentic_knowledgebase:
            if should_stream and overrides.get("use_web_source"):
                raise Exception(
                    "Streaming is not supported with agentic retrieval when web source is enabled. Please disable streaming or web source."
                )
            extra_info = await self.run_agentic_retrieval_approach(messages, overrides, auth_claims)
        else:
            extra_info = await self.run_search_approach(messages, overrides, auth_claims)

        if extra_info.answer:
            # If agentic retrieval already provided an answer, skip final call to LLM
            async def return_answer() -> Response:
                return Response(
                    id="no-final-call",
                    object="response",
                    parallel_tool_calls=True,
                    tool_choice="auto",
                    tools=[],
                    created_at=0,
                    model=self.chatgpt_model,
                    output=[
                        ResponseOutputMessage(
                            id="msg-no-final-call",
                            type="message",
                            role="assistant",
                            status="completed",
                            content=[{"type": "output_text", "text": extra_info.answer, "annotations": []}],  # type: ignore[list-item]
                        )
                    ],
                    status="completed",
                )

            return (extra_info, return_answer())

        messages = self.prompt_manager.build_conversation(
            system_template_path="chat_answer.system.jinja2",
            system_template_variables=self.get_system_prompt_variables(overrides.get("prompt_template"))
            | {
                "include_follow_up_questions": bool(overrides.get("suggest_followup_questions")),
                "image_sources": extra_info.data_points.images,
                "citations": extra_info.data_points.citations,
            },
            user_template_path="chat_answer.user.jinja2",
            user_template_variables={
                "user_query": original_user_query,
                "text_sources": extra_info.data_points.text,
            },
            user_image_sources=extra_info.data_points.images,
            past_messages=messages[:-1],
        )

        response_coroutine = cast(
            Awaitable[Response] | Awaitable[AsyncStream[ResponseStreamEvent]],
            self.create_response(
                self.chatgpt_deployment,
                self.chatgpt_model,
                messages,
                overrides,
                self.get_response_token_limit(self.chatgpt_model, self.RESPONSE_DEFAULT_TOKEN_LIMIT),
                should_stream,
            ),
        )
        extra_info.thoughts.append(
            self.format_thought_step_for_chatcompletion(
                title="Prompt to generate answer",
                messages=messages,
                overrides=overrides,
                model=self.chatgpt_model,
                deployment=self.chatgpt_deployment,
                usage=None,
            )
        )
        return (extra_info, response_coroutine)

    async def run_search_approach(
        self, messages: list[EasyInputMessageParam], overrides: dict[str, Any], auth_claims: dict[str, Any]
    ):
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        use_query_rewriting = True if overrides.get("query_rewriting") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        search_index_filter = self.build_filter(overrides)
        access_token = auth_claims.get("access_token")
        send_text_sources = overrides.get("send_text_sources", True)
        send_image_sources = overrides.get("send_image_sources", self.multimodal_enabled) and self.multimodal_enabled
        search_text_embeddings = overrides.get("search_text_embeddings", True)
        search_image_embeddings = (
            overrides.get("search_image_embeddings", self.multimodal_enabled) and self.multimodal_enabled
        )

        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question

        rewrite_result = await self.rewrite_query(
            prompt_template="query_rewrite.system.jinja2",
            prompt_variables={
                "user_query": original_user_query,
                "past_messages": messages[:-1],
            },
            overrides=overrides,
            chatgpt_model=self.chatgpt_model,
            chatgpt_deployment=self.chatgpt_deployment,
            user_query=original_user_query,
            response_token_limit=self.get_response_token_limit(
                self.chatgpt_model, 100
            ),  # Setting too low risks malformed JSON, setting too high may affect performance
            tools=self.query_rewrite_tools,
            temperature=0.0,  # Minimize creativity for search query generation
            no_response_token=self.NO_RESPONSE,
        )

        query_text = rewrite_result.query

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

        vectors: list[VectorQuery] = []
        if use_vector_search:
            if search_text_embeddings:
                vectors.append(await self.compute_text_embedding(query_text))
            if search_image_embeddings:
                vectors.append(await self.compute_multimodal_embedding(query_text))

        results = await self.search(
            top,
            query_text,
            search_index_filter,
            vectors,
            use_text_search,
            use_vector_search,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
            use_query_rewriting,
            access_token,
        )

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        data_points = await self.get_sources_content(
            results,
            use_semantic_captions,
            include_text_sources=send_text_sources,
            download_image_sources=send_image_sources,
            user_oid=auth_claims.get("oid"),
        )
        extra_info = ExtraInfo(
            data_points,
            thoughts=[
                self.format_thought_step_for_chatcompletion(
                    title="Prompt to generate search query",
                    messages=rewrite_result.messages,
                    overrides=overrides,
                    model=self.chatgpt_model,
                    deployment=self.chatgpt_deployment,
                    usage=rewrite_result.completion.usage,
                    reasoning_effort=rewrite_result.reasoning_effort,
                ),
                ThoughtStep(
                    "Search using generated search query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "use_query_rewriting": use_query_rewriting,
                        "top": top,
                        "filter": search_index_filter,
                        "use_vector_search": use_vector_search,
                        "use_text_search": use_text_search,
                        "search_text_embeddings": search_text_embeddings,
                        "search_image_embeddings": search_image_embeddings,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
            ],
        )
        return extra_info

    async def run_agentic_retrieval_approach(
        self,
        messages: list[EasyInputMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
    ):
        search_index_filter = self.build_filter(overrides)
        access_token = auth_claims.get("access_token")
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0)
        send_text_sources = overrides.get("send_text_sources", True)
        send_image_sources = overrides.get("send_image_sources", self.multimodal_enabled) and self.multimodal_enabled
        retrieval_reasoning_effort = overrides.get("retrieval_reasoning_effort", self.retrieval_reasoning_effort)
        # Overrides can only disable web source support configured at construction time.
        use_web_source = self.web_source_enabled
        override_use_web_source = overrides.get("use_web_source")
        if isinstance(override_use_web_source, bool):
            use_web_source = use_web_source and override_use_web_source
        # Overrides can only disable sharepoint source support configured at construction time.
        use_sharepoint_source = self.use_sharepoint_source
        override_use_sharepoint_source = overrides.get("use_sharepoint_source")
        if isinstance(override_use_sharepoint_source, bool):
            use_sharepoint_source = use_sharepoint_source and override_use_sharepoint_source
        if use_web_source and retrieval_reasoning_effort == "minimal":
            raise Exception("Web source cannot be used with minimal retrieval reasoning effort.")

        selected_client, effective_web_source, effective_sharepoint_source = self._select_knowledgebase_client(
            use_web_source,
            use_sharepoint_source,
        )

        agentic_results = await self.run_agentic_retrieval(
            messages=messages,
            knowledgebase_client=selected_client,
            search_index_name=self.search_index_name,
            filter_add_on=search_index_filter,
            minimum_reranker_score=minimum_reranker_score,
            access_token=access_token,
            use_web_source=effective_web_source,
            use_sharepoint_source=effective_sharepoint_source,
            retrieval_reasoning_effort=retrieval_reasoning_effort,
        )

        data_points = await self.get_sources_content(
            agentic_results.documents,
            use_semantic_captions=False,
            include_text_sources=send_text_sources,
            download_image_sources=send_image_sources,
            user_oid=auth_claims.get("oid"),
            web_results=agentic_results.web_results,
            sharepoint_results=agentic_results.sharepoint_results,
        )

        return ExtraInfo(
            data_points,
            thoughts=agentic_results.thoughts,
            answer=agentic_results.answer,
        )

    def _select_knowledgebase_client(
        self,
        use_web_source: bool,
        use_sharepoint_source: bool,
    ) -> tuple[KnowledgeBaseRetrievalClient, bool, bool]:
        if use_web_source and use_sharepoint_source:
            if self.knowledgebase_client_with_web_and_sharepoint:
                return self.knowledgebase_client_with_web_and_sharepoint, True, True
            if self.knowledgebase_client_with_web:
                return self.knowledgebase_client_with_web, True, False
            if self.knowledgebase_client_with_sharepoint:
                return self.knowledgebase_client_with_sharepoint, False, True

        if use_web_source and self.knowledgebase_client_with_web:
            return self.knowledgebase_client_with_web, True, False

        if use_sharepoint_source and self.knowledgebase_client_with_sharepoint:
            return self.knowledgebase_client_with_sharepoint, False, True

        if self.knowledgebase_client:
            return self.knowledgebase_client, False, False
        raise ValueError("Agentic retrieval requested but no knowledge base is configured")

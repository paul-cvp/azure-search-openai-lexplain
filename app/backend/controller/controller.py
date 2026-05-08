# Controller class with ask, chat, and orchestrate methods
from approaches.approach import Approach
from typing import Any, cast
from quart import current_app

async def ask(request_json, context, config, *args, **kwargs):
    approach: Approach = cast(Approach, config)
    r = await approach.run(
        request_json["messages"], context=context, session_state=request_json.get("session_state")
    )
    return r

async def orchestrate(request_json, context, config, *args, **kwargs):
    approach: Approach = cast(Approach, config)
    r = await approach.run(
        request_json["messages"], context=context, session_state=request_json.get("session_state")
    )
    return r

async def chat(*args, **kwargs):
    pass


import asyncio

from agentic_copilot.models.utils.agents_util import AgentsState
from agentic_copilot.workflows.workflow import CopilotFlow


async def main():
    user_id = input("\033[34mHi, Cold you give me your user_id?\033[0m\t")
    state = AgentsState(user_id=user_id)
    utterance = input("\033[34mThanks, How can I help you?\033[0m\t")
    conv_going = True
    conv_continue = False

    while conv_going and utterance != "STOP":
        workflow = CopilotFlow(timeout=300)
        result, state = await workflow.run(state=state, utterance=utterance, continue_bool=conv_continue)

        utterance = input(f"\033[34m{result}\033[0m\n\n")

        if utterance == "STOP":
            conv_going = False

        conv_continue = True


loop = asyncio.get_event_loop()
loop.run_until_complete(main())

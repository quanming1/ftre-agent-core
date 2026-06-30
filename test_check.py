from ftre_agent_core.agent.event import EventType, AssistantMessageEvent, AssistantMessageCompleteEvent, UserMessageEvent, ToolCallEvent, DoneReason
print('EventType values:', [e.value for e in EventType])
print('AssistantMessageEvent type:', AssistantMessageEvent(content='x').type)
print('AssistantMessageCompleteEvent type:', AssistantMessageCompleteEvent(content='x').type)
print('UserMessageEvent type:', UserMessageEvent(content='x').type)
print('ToolCallEvent fields:', ToolCallEvent(tool_id='a', tool_name='b', arguments={})._data_dict())
print('DoneReason values:', [e.value for e in DoneReason])

# OpenAI Cheat Sheet
OpenAI Cheat Sheet with the most needed stuff..




## Chat-GPT
- https://platform.openai.com/docs/api-reference/chat/create

<br><br>

### Node.js (https://www.npmjs.com/package/openai)
```
import { Configuration, OpenAIApi } from "openai"

const configuration = new Configuration({
  apiKey: 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
})

const openai = new OpenAIApi(configuration)
// const engines = await openai.listEngines()
// console.log(engines)

const content = `
What is 1+1?
`

const messages = [{"role": "user", content}]
const model = 'gpt-3.5-turbo'

const completion = await openai.createChatCompletion({
  model,
  messages,
})

const parsedResponse = JSON.parse(completion.data.choices[0].message.content)
console.log(JSON.stringify(parsedResponse, null,4))
```

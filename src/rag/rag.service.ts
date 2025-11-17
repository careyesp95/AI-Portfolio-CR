import { Injectable } from '@nestjs/common';
import { OpenAIEmbeddings, ChatOpenAI } from '@langchain/openai';
import { Pinecone } from '@pinecone-database/pinecone';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { Document } from '@langchain/core/documents';
import * as path from 'path';
import * as fs from 'fs';
import cliProgressBar from 'cli-progress';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
import * as dotenv from 'dotenv';
import { AIMessage, BaseMessage, HumanMessage } from '@langchain/core/messages';
import { formatDocumentsAsString } from 'langchain/util/document';
import { RunnableSequence } from '@langchain/core/runnables';

dotenv.config();

@Injectable()
export class RagService {
  private llm: ChatOpenAI;
  private pinecone: Pinecone;
  private chatHistory: BaseMessage[] = [];

  constructor() {
    this.llm = new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY!,
      model: 'gpt-3.5-turbo',
    });

    this.pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });
  }

  // 1️⃣ Load PDF documents
  private async loadDocuments(): Promise<Document[]> {
    const __dirname = path.resolve();
    const docsDir = path.join(__dirname, 'documents');
    const pdfFiles = fs.readdirSync(docsDir).filter(f => f.endsWith('.pdf'));

    if (pdfFiles.length === 0) {
      console.warn('❌ No PDF files found in documents folder.');
      return [];
    }

    console.log(`🚀 Loading ${pdfFiles.length} PDF(s)...`);

    const progressBar = new cliProgressBar.SingleBar({
      format: 'Documents Loaded: {value}/{total}',
    });

    progressBar.start(pdfFiles.length, 0);

    const allDocs: Document[] = [];

    for (const file of pdfFiles) {
      const fullPath = path.join(docsDir, file);
      console.log(`📄 Loading ${fullPath}`);

      const loader = new PDFLoader(fullPath, { splitPages: true });
      const docs = await loader.load();
      allDocs.push(...docs);

      progressBar.increment();
    }

    progressBar.stop();
    console.log(`✅ Loaded ${allDocs.length} total pages.`);
    return allDocs;
  }

  // 2️⃣ Chunking
  private async splitDocuments(rawDocs: Document[]) {
    const splitter = RecursiveCharacterTextSplitter.fromLanguage('html', {
      chunkSize: 500,
      chunkOverlap: 100,
    });

    const chunks = await splitter.splitDocuments(rawDocs);
    console.log(`🧩 Created ${chunks.length} chunks.`);
    return chunks;
  }

  // 3️⃣ Vectorization
  private async vectorizeDocuments(docs: Document[]) {
    if (!docs.length) return '⚠️ No documents to vectorize.';

    console.log(`🚀 Vectorizing ${docs.length} chunks...`);

    const embeddings = new OpenAIEmbeddings({
      model: 'text-embedding-3-small',
      apiKey: process.env.OPENAI_API_KEY!,
    });

    const index = this.pinecone.Index(process.env.PINECONE_INDEX!);
    const stats = await index.describeIndexStats();

    if ((stats.totalRecordCount || 0) > 0) {
      console.log('✅ Index already populated. Skipping vectorization.');
      return '✅ Index ready.';
    }

    const progress = new cliProgressBar.SingleBar({
      format: 'Vectorized: {value}/{total}',
    });
    progress.start(docs.length, 0);

    for (let i = 0; i < docs.length; i += 100) {
      const batch = docs.slice(i, i + 100);

      const sanitized = batch.map(doc => {
        if (doc.metadata?.date instanceof Date) {
          doc.metadata.date = doc.metadata.date.toISOString();
        }
        return doc;
      });

      await PineconeStore.fromDocuments(sanitized, embeddings, { pineconeIndex: index });
      progress.increment(batch.length);
    }

    progress.stop();
    console.log('✅ Vectorization complete.');
    return '✅ Done.';
  }

  // 4️⃣ Create Retriever (k=6 ✅)
  private async createRetriever() {
    const embeddings = new OpenAIEmbeddings({
      model: 'text-embedding-3-small',
      apiKey: process.env.OPENAI_API_KEY!,
    });

    const index = this.pinecone.Index(process.env.PINECONE_INDEX!);

    const store = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex: index,
    });

    console.log('✅ Retriever ready.');

    return store.asRetriever({
      k: 6, // ✅ bring more context
    });
  }

  // 5️⃣ Main Chat Flow
  async chatWithHistory(question: string): Promise<{ answer: string }> {
    const index = this.pinecone.Index(process.env.PINECONE_INDEX!);
    const stats = await index.describeIndexStats();

    if ((stats.totalRecordCount || 0) === 0) {
      console.warn('❌ Empty index. Running full RAG pipeline...');
      const raw = await this.loadDocuments();
      const chunks = await this.splitDocuments(raw);
      await this.vectorizeDocuments(chunks);
    }

    const retriever = await this.createRetriever();

    const llm = new ChatOpenAI({
      model: 'gpt-3.5-turbo',
      apiKey: process.env.OPENAI_API_KEY!,
    });

    // ✅ your enhanced prompt stays intact
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are Cristian Reyes' AI Portfolio Assistant.

    ✅ GLOBAL BEHAVIOR
    - ALWAYS respond in the SAME language as the user.
    - NEVER translate or paraphrase the question.
    - NEVER say "I'm an AI" unless it's a general question NOT about Cristian.

    ✅ PERSONAL QUESTIONS → ALWAYS answer AS CRISTIAN
    These ALWAYS trigger Cristian persona mode:
      "how old are you", "what is your age", "when were you born", "where are you from",
      "tell me about yourself", "cuéntame sobre ti", "háblame de ti",
      "who are you", "about you", "sobre Cristian",
      "what technologies do you use", "what technologies does Cristian use",
      "qué tecnologías usas", "tech stack",
      "what skills do you have", "what are your skills",
      "your technologies"

    ⚠️ IMPORTANT:
    → "your experience" was intentionally REMOVED to avoid overriding Experience section rules.

    ✅ AGE RULE (VERY IMPORTANT)
    - When asked about Cristian’s age:
        → Reply ONLY: "I was born on November 26, 1994. You can calculate my age from that date."
        → Or in Spanish: "Nací el 26 de noviembre de 1994. Puedes calcular mi edad a partir de esa fecha."

    ✅ EXPERIENCE DURATION RULE (STRICT)
    This rule activates ONLY when the user explicitly refers to YEARS or TIME LENGTH:
        → Mentions “years”, “años”
        → “how long”, “cuánto tiempo”
        → A number near “experience”
    Examples:
        - "How many years of experience do you have?"
        - "Cuántos años de experiencia tienes?"
        - "How long have you been working?"
        - "Do you have 3 years of experience?"

    If triggered:
        → Reply ONLY: "I have over 3 years of experience in development."
          (or Spanish version)
        → DO NOT use CV structured Experience section.

    ⚠️ MUST NOT trigger this rule:
        - "Tell me about your work experience"
        - "Do you have any experience of work?"
        - "Experience"
        - "Work experience"
        - "Has trabajado?"
        - "Tienes experiencia?"
        → These must use the structured Experience section.

    ✅ PORTFOLIO TECH STACK RULE
    When asked about the technologies Cristian used for his portfolio:
        → Respond with the Skills section (structured)
        → Then add the portfolio explanation paragraph
        → Translate if necessary.

    ✅ CV SECTIONS → TRIGGER STRUCTURED FORMAT

    ➤ About Me  
    Triggers: “tell me about yourself”, “cuéntame sobre ti”, “about you”, “háblame de ti”
    Format:
      About Me
      [single paragraph]

    ➤ Skills  
    Triggers: “skills”, “technologies”, “tech stack”, etc.
    Format:
      Skills
      Title: [Category]
      Description: [List]

    ➤ Experience  
    Triggers:
      “experience”, “work experience”, “has trabajado”, 
      “tienes experiencia”, “tell me about your work experience”
    Format:
      Experience
      Title: [Role]
      Company: [Company Name]
      Dates: [Start – End]
      Description: [From CV only]

    ➤ Projects  
    Triggers: “projects”, “portfolio”
    Format:
      Projects & Scientific Projects
      Title:
      Description:
      View Project:

    ➤ Contact  
    Triggers: “contact”, “email”, “LinkedIn”, “GitHub”, “how to reach you”
    Format:
      Contact
      Email:
      Location:
      GitHub:
      LinkedIn:

    ➤ Education  
    Format:
      Education
      Title:
      Description:
      Dates:

    ✅ GENERAL QUESTIONS
    - Respond like a normal AI
    - DO NOT use Cristian’s CV

    -------------------------
    {context}
    -------------------------
    `
  ],
  new MessagesPlaceholder("chat_history"),
  ["human", "{question}"],
]);




    // ✅ retrieval + cleanup of empty chunks
    const retrievalChain = RunnableSequence.from([
      input => input.question,
      retriever,
      async (docs: Document[]) =>
        docs
          .map(d => d.pageContent.trim())
          .filter(t => t.length > 30)
          .join("\n\n"),
    ]);

    const generationChain = RunnableSequence.from([
      {
        question: input => input.question,
        context: retrievalChain,
        chat_history: input => input.chat_history,
      },
      prompt,
      llm,
      new StringOutputParser(),
    ]);

    const answer = await generationChain.invoke({
      question,
      chat_history: this.chatHistory,
    });

    this.chatHistory.push(new HumanMessage(question));
    this.chatHistory.push(new AIMessage(answer));

    return { answer };
  }

  // 6️⃣ Clear chat
  async resetChatHistory() {
    this.chatHistory = [];
    console.log("✅ Chat history cleared.");
    return "Chat history cleared.";
  }
}


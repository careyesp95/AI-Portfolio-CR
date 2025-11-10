import { Controller, Post, Body, Get } from '@nestjs/common';
import { RagService } from './rag.service';
import type { Response } from 'express'; 
import { Res } from '@nestjs/common';


@Controller('api')
export class RagController {
  constructor(private readonly ragService: RagService) {}

  @Post('portfolio/ask-me')
  async ask(
    @Body('question') question: string,
  ) {

    if (!question) {
      return { answer: 'Please provide a question.' };
    }
    const { answer } = await this.ragService.chatWithHistory(question);
    return { answer };
  }

  @Get('clear-chat')
  async clearChat() {
    const message = await this.ragService.resetChatHistory();
    return { message };
  }

}

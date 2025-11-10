import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { RagController } from './rag/rag.controller';
import { RagService } from './rag/rag.service';

@Module({
  imports: [],
  controllers: [AppController, RagController],
  providers: [AppService, RagService],
})
export class AppModule {}

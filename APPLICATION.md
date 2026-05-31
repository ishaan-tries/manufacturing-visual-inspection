# Why I Built It and How I Use AI Coding Tools

## What I Built

Two repos:

- github.com/ishaan-tries/manufacturing-rag-assistant
  LangChain, FAISS, Gemini API. Ingests technical PDFs, answers questions with source citations.

- github.com/ishaan-tries/manufacturing-visual-inspection
  PyTorch, ResNet18, MVTec Anomaly Detection dataset. 94.92% validation accuracy.
  Weighted CrossEntropyLoss for class imbalance, CUDA GPU training, OpenCV real-time inference.

I use Claude Code as my primary development environment. Not as autocomplete. As a thinking partner
that I argue with.

---

## The Six Questions

**Agency. What's the last thing nobody asked you to do but you did it?**

The RAG evaluation layer. Nobody asked for it.

Here is the problem I saw: companies in Germany are migrating old data from SAP and legacy systems
into new AI-powered platforms. They want to build RAG on top of it. But they do not know what is
clean data and what is dirty data. They just dump everything in and wonder why the answers are bad.

So I built an evaluation layer that scores the corpus before the LLM ever sees it. Lexical diversity
(TTR, MTLD), readability (Flesch-Kincaid, Gunning Fog), topic coherence. It tells you how good your
data is, not just how good your model is. Nobody told me to build it. I saw the problem and started.

**See Problems. How do you act?**

I start working. I make a rough plan, then I keep updating it as reality pushes back.
The plan is never right the first time. The starting is what matters.

**Obsession to learn. What has been the last thing you learned?**

Hexagonal architecture. Also called ports and adapters.

I am rebuilding my RAG pipeline so the core domain logic is completely isolated from the LLM,
the vector store, and the embedding model. The idea is simple: the core should not care whether
I am using Gemini or Claude or FAISS or ChromaDB. Those are infrastructure details. The domain
logic stays clean regardless of what changes around it.

I learned this because I kept breaking my own pipeline when I switched LLM providers.
That is a design problem, not a code problem. So I fixed the design.

**Understand yourself as first customer. How are you testing things?**

I run it on real data and see if the output is actually useful to me.

Not toy examples. The defect detector was tested on the actual MVTec industrial benchmark,
not synthetic images. The RAG system was tested on real automotive technical manuals.
If I would not use the output to make a decision, it is not good enough yet.

**Eye for good products. What product inspires you and why?**

Technology as a direction, not a single product.

We are moving faster than most people realise. Five years ago the idea of generating coherent
video from text was science fiction. Now it is Tuesday. LLMs, robots, video generation, IoT.
The thing that genuinely occupies my thinking is the feedback loop: AI generates more data,
more AI trains on that data, some of it is noise, but the signal keeps getting stronger anyway.
Whether that leads to singularity or just to increasingly capable narrow systems, I do not know.
But the trajectory is undeniable and I want to be building things on that curve, not watching it.

**This is new. No years required, but what makes you different?**

I build the thing that exposes the problem nobody else noticed yet.

Everyone builds RAG. I built the layer that tells you whether your RAG corpus deserves to exist.
Everyone uses Claude Code. I use it to argue about architecture at 11pm and then rebuild the whole
thing because the argument was right.

I am a Master's student in Data Science and AI at Deggendorf Institute of Technology in Germany.
Two peer-reviewed publications in applied ML and NLP. I ship things and I break them and I fix them.

---

## Secret Hacks for Teaching AI Coding Tools

1. Tell Claude what is wrong with its own suggestion before asking for a better one.
   It argues back and usually the argument is the real answer.

2. Paste the error first, then the code. Not the other way around.
   Context before content.

3. Ask for the architecture decision, not the implementation.
   The implementation you can figure out. The architecture decision is where the thinking happens.

4. When something feels too easy, it is wrong. Push back.

5. Use it to rubber duck. Say the problem out loud in the prompt.
   Half the time you solve it while writing the question.

# Yori Chat - iMessage Style Interface

A Next.js 14 application with an iMessage-style chat interface for chatting with Yori.

## Getting Started

First, install the dependencies:

```bash
npm install
```

Then, run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Features

- 📱 iMessage-style chat interface
- 💬 Real-time typing indicators with bouncing dots
- 🔄 Integration with Yori API at `http://localhost:8000/chat`
- 🎨 Beautiful UI with TailwindCSS
- ⚡ Built with Next.js 14 App Router
- 🔤 Full TypeScript support

## Project Structure

```
frontend-next/
├── src/
│   ├── app/
│   │   ├── layout.tsx         # Root layout
│   │   ├── page.tsx           # Main chat interface
│   │   └── globals.css        # Global styles
│   └── components/
│       └── TypingIndicator.tsx # Animated typing dots
├── tailwind.config.ts         # Tailwind configuration
├── tsconfig.json             # TypeScript configuration
└── package.json              # Dependencies and scripts
```

## API Integration

The chat interface sends POST requests to `http://localhost:8000/chat` with the following format:

```json
{
  "message": "user message",
  "user_id": "test_user"
}
```

Expected response format:

```json
{
  "message": "Yori's response"
}
```

## Chat Behavior

1. User types a message and hits Send
2. User's message appears immediately as a blue bubble on the right
3. After 800ms, a typing indicator appears as a grey bubble on the left
4. API request is sent to the backend
5. When response arrives, typing indicator is replaced with Yori's message

## Technologies Used

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **TailwindCSS** - Utility-first CSS framework
- **React Hooks** - State management
- **ESLint** - Code linting
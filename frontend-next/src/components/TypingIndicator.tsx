export default function TypingIndicator() {
  return (
    <div className="flex items-center space-x-1">
      <div className="flex space-x-1">
        <div 
          className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
          style={{ animationDelay: '0ms', animationDuration: '1.4s' }}
        />
        <div 
          className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
          style={{ animationDelay: '0.2s', animationDuration: '1.4s' }}
        />
        <div 
          className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"
          style={{ animationDelay: '0.4s', animationDuration: '1.4s' }}
        />
      </div>
    </div>
  )
}
import { redirect } from "next/navigation";

// Root → redirect to explorer
export default function HomePage() {
  redirect("/explorer");
}
